"""
This module provides the NADQ algorithm

Aligned- and Non-aligned-double-JPEG-compression-based detector.

Algorithm attribution:
T.Bianchi, A.Piva, "Image Forgery Localization via Block-Grained
Analysis of JPEG Artifacts",  IEEE Transactions on Information Forensics &
Security, vol. 7,  no. 3,  June 2012,  pp. 1003 - 1017.

Based on code from:
Zampoglou, M., Papadopoulos, S., & Kompatsiaris, Y. (2017). Large-scale evaluation of splicing localization algorithms for web images. Multimedia Tools and Applications, 76(4), 4801–4834.
"""

from pyIFD.util import ibdct, jpeg_rec, bdct, dequantize, ibdct
from scipy.signal import convolve2d
from scipy.ndimage import correlate
from scipy.fft import idct
import cupy as cp
import jpegio as jio
import math
from scipy.signal import convolve2d
from scipy.ndimage import correlate
from scipy.fft import idct
from scipy.signal import fftconvolve

def NADQ(impath):
    """
    Main driver for NADQ algorithm
    Args:
        impath: Icput image path
    Returns:
        OutputMap: OutputMap
    """
    if impath[-4:] == ".jpg":
        try:
            OutputMap = getJmapNA_EM(jio.read(impath))
        except Exception as e:
            print('JPEGIO exception: ' + str(e))
            return
    else:
        print('Only .jpg supported')
    return OutputMap

# JPEG_QTABLE  Generate standard JPEG quantization tables
#
#   T=JPEG_QTABLE(QUALITY,TNUM,FORCE_BASELINE)
#
#   Returns a quantization table T given in JPEG spec, section K.1 and scaled
#   using a quality factor.  The scaling method used is the same as that used
#   by the IJG (Independent JPEG Group) code library.
#
#   QUALITY values should range from 1 (terrible) to 100 (very good), the
#   scale recommended by IJG.  Default is 50, which represents the tables
#   defined by the standard used without scaling.
#
#   TNUM should be a valid table number, either 0 (used primarily for
#   luminance channels), or 1 (used for chromatic channels). Default is 0.
#
#   FORCE_BASELINE clamps the quantization table entries to have values
#   between 1..255 to ensure baseline compatibility with all JPEG decoders.
#   By default, values are clamped to a range between 1..32767.  These are
#   the same ranges used by the IJG code library for generating standard
#   quantization tables.


def jpeg_qtable(quality=50, tnum=0, force_baseline=0):

    # convert to linear quality scale
    if (quality <= 0):
        quality = 1
    if (quality > 100):
        quality = 100
    if (quality < 50):
        quality = 5000 / quality
    else:
        quality = 200 - quality*2

    if tnum == 0:
        # This is table 0 (the luminance table):
        t = [16, 11, 10, 16, 24, 40, 51, 61,
             12, 12, 14, 19, 26, 58, 60, 55,
             14, 13, 16, 24, 40, 57, 69, 56,
             14, 17, 22, 29, 51, 87, 80, 62,
             18, 22, 37, 56, 68, 109, 103, 77,
             24, 35, 55, 64, 81, 104, 113, 92,
             49, 64, 78, 87, 103, 121, 120, 101,
             72, 92, 95, 98, 112, 100, 103, 99]

    elif tnum == 1:
        # This is table 1 (the chrominance table):
        t = [17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
             47, 66, 99, 99, 99, 99, 99, 99,
             99, 99, 99, 99, 99, 99, 99, 99,
             99, 99, 99, 99, 99, 99, 99, 99,
             99, 99, 99, 99, 99, 99, 99, 99,
             99, 99, 99, 99, 99, 99, 99, 99]

    t = cp.reshape(t,(8,8),order='F').T
    t = cp.floor((t * quality + 50)/100)
    t[t < 1] = 1

    t[t > 32767] = 32767  # max quantizer needed for 12 bits
    if (force_baseline):
        t[t > 255] = 255

    return t


def LLR(x, nz, Q, phase, sig):
    binHist=range(-2**11, 2**11)
    center=2**11
    # Finished review
    w = int(cp.ceil(3*sig))
    k = list(range(-w,w+1))
    g = cp.array([math.exp(-kk**2/sig**2/2) for kk in k])
    g = g/cp.sum(g)
    N = cp.size(x) / cp.size(binHist)

    bppm = cp.zeros(cp.shape(binHist))
    bppm[center + phase::Q] = Q
    bppm[center + phase::-Q] = Q
    bppm = cp.convolve(g, bppm)
    bppm = bppm[w:-w]
    bppm = (bppm*N + 1)
    LLRmap = cp.log(bppm / cp.mean(bppm))
    LLRmap[center] = nz * LLRmap[center]
    x=cp.round(x).astype("int")+center
    def lmap(xx):
        return LLRmap[xx]
    vlmap=cp.vectorize(lmap)
    L = vlmap(x)
    return L


def EMperiod(x, Qmin, Qmax, alpha0, h0, dLmin, maxIter, hcal, bias, sig):
    # Finished review
    Qvec = list(range(int(Qmin),int(Qmax)+1))
    alphavec = alpha0*cp.ones(cp.shape(Qvec))
    h1mat = cp.zeros((len(Qvec), len(x)))
    for k in range(len(Qvec)):
        h1mat[k,:] = h1period(x, Qvec[k], hcal, bias, sig)
    Lvec = cp.ones(cp.shape(Qvec))*float('-inf')
    Lmax = float('-inf')
    delta_L = float('inf')
    ii = 0
        # Markos: for cases where the if clause is never activated    
    Q=Qvec[0]
    alpha=alphavec[0]

    while delta_L > dLmin and ii < maxIter:
        ii +=1

        for k in range(len(Qvec)):
            # expectation
            beta0 = h0*alphavec[k] / (h0*alphavec[k] + h1mat[k,:]*(1 - alphavec[k]))
            # maximization
            alphavec[k] = cp.mean(beta0)
            # compute true log-likelihood of mixture
            L = cp.sum(cp.log(alphavec[k]*h0 + (1-alphavec[k])*h1mat[k,:]))
            if (L > Lmax):
                Lmax = L
                Q = Qvec[k]
                alpha = alphavec[k]
                if (L - Lvec[k] < delta_L):
                    delta_L = L - Lvec[k]
            Lvec[k] = L
    return [Q, alpha, Lmax]

def h1period(x, Q, hcal, bias, sig):
    #Check h1 period first
    binHist=range(-2**11,2**11)
    center=2**11
    #Finished review
    N = cp.sum(hcal)
    # simulate quantization
    if Q % 2 == 0:
        hs = cp.ones(Q-1)
        hs=cp.append(hs,0.5)
        hs=cp.insert(hs,0, 0.5)
        ws = int(Q/2)
    else:
        hs = cp.ones(Q)
        ws = int((Q-1)/2)
    h2 = cp.convolve(hcal,hs)
    # simulate dequantization
    h1 = cp.zeros(cp.shape(binHist))
    h1[center::Q] = h2[center + ws:-ws:Q]
    h1[center::-Q] = h2[center + ws:ws-1:-Q]
    # simulate rounding/truncation
    w = int(cp.ceil(3*sig))
    k = range(-w,w+1)
    g = [math.exp(-(kk+bias)**2/sig**2/2) for kk in k]
    h1 = cp.convolve(h1, g)
    h1 = h1[w:-w]
    # normalize probability and use Laplace correction to avoid p1 = 0
    h1 /= sum(h1)
    h1 = (h1*N+1)/(N+cp.size(binHist))
    x=cp.array(x)
    p1=cp.take(h1,cp.round(cp.nextafter(x,x+1)).astype("int")+center)
    return p1


def getJmapNA_EM(image, ncomp=1, c2=6):
    """
    Detects and localizes tampered areas in double compressed JPEG images.

    Args:
        image: JPEG object TODO: Change to impath
        ncomp: index of color component (1 = Y, 2 = Cb, 3 = Cr)
        c2: number of DCT coefficients to consider (1 <= c2 <= 64)
        ncomp:
        c2:

    Returns:
        LLRmap(:,:,c): estimated likelihood of being doubly compressed for each 8x8 image block
                       using standard model and c-th DCT frequency (zig-zag order)
        LLRmap_s(:,:,c): estimated likelihood of being doubly compressed for each 8x8 image block
                         using simplified model and c-th DCT frequency (zig-zag order)
        k1e: estimated shift of first compression
        k2e: estimated shift of second compression TODO: ?
        alphatable: mixture parameter for each DCT frequency
    """
    coeffArray = image.coef_arrays[ncomp-1]
    qtable = image.quant_tables[image.comp_info[ncomp-1].quant_tbl_no]
    q1table = cp.ones((8,8))
    minQ = cp.maximum(2,cp.floor(qtable/cp.sqrt(3)))
    maxQ = cp.maximum(jpeg_qtable(50),qtable)
    # estimate rounding and truncation error
    Im = jpeg_rec(image)[0]
    ImTmp = Im.copy()
    ImTmp=cp.maximum(0,ImTmp)
    ImTmp[ImTmp > 255] = 255
    E = Im - cp.round(ImTmp)
    Edct = bdct(0.299 * E[:, :, 0] + 0.587 * E[:, :, 1] + 0.114 * E[:, :, 2])

    # compute DCT coeffs of decompressed image
    Im = ibdct(dequantize(coeffArray, qtable))
    coeff = [1, 9, 2, 3, 10, 17, 25, 18, 11, 4, 5, 12, 19, 26, 33, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 57, 50, 43, 36, 29, 22, 15, 8, 16, 23, 30, 37, 44, 51, 58, 59, 52,
             45, 38, 31, 24, 32, 39, 46, 53, 60, 61, 54, 47, 40, 48, 55, 62, 63, 56, 64]
    center = 2**11

    B = cp.ones((8,8))/8
    DC = cp.rot90(convolve2d(cp.rot90(Im, 2), cp.rot90(B, 2)), 2)
    DC = DC[7:, 7:]
    EDC = Edct[::8, ::8]
    varE = cp.var(EDC)
    bias = cp.mean(EDC)
    sig = cp.sqrt(qtable[0, 0]**2 / 12 + varE)
    alphatable = cp.ones((8,8))
    Ims=cp.shape(Im)
    LLRmap = cp.zeros((int(Ims[0]/8), int(Ims[1]/8), c2))
    LLRmap_s = cp.zeros((int(Ims[0]/8), int(Ims[1]/8), c2))
    k1e = 1
    k2e = 1
    Lmax = -cp.inf
    # estimate shift of first compression
    for k1 in range(8):
        for k2 in range(8):
            binHist = range(-2**11, 2**11)
            if (k1 + 1 > 1 or k2 + 1 > 1):
                DCpoly = DC[k1::8, k2::8]
                # choose shift for estimating unquantized distribution through
                # calibration
                if k1 < 4:
                    k1cal = k1 + 2
                else:
                    k1cal = k1
                if k2 < 4:
                    k2cal = k2 + 2
                else:
                    k2cal = k2
                DCcal = DC[k1cal-1::8, k2cal-1::8]
                binHist = cp.arange(-2**11, 2**11-1)+0.5
                binHist = cp.append(binHist, max(2**11, cp.max(DCcal)))
                binHist = cp.insert(binHist, 0, min(-2**11, cp.min(DCcal)))
                hcal = cp.histogram(DCcal, binHist)[0]
                hcalnorm = (hcal+1)/(cp.size(DCcal)+cp.size(binHist)-1)
                # define mixture components
                h0=cp.array(cp.take(hcalnorm,cp.round(cp.ndarray.flatten(DCpoly,order='F')).astype("int")+center))
                # estimate parameters of first compression
                [Q, alpha, L] = EMperiod(cp.ndarray.flatten(DCpoly,order='F'), minQ[0, 0], maxQ[0, 0], 0.95, h0, 5, 20, hcal, bias, sig)
                if L > Lmax:
                    # simplified model
                    nz = cp.count_nonzero(DCpoly)/cp.size(DCpoly)
                    LLRmap_s[:, :, 0] = LLR(DCpoly, nz, Q, int(cp.round(bias)), sig)
                    # standard model
                    ppu = cp.log(cp.divide(h1period(range(-2**11,2**11), Q, hcal, bias, sig),cp.take(hcalnorm,range(2**12))))
                    DCpoly=cp.round(DCpoly).astype("int")+center
                    def pmap(xx):
                        return ppu[xx]
                    vpmap=cp.vectorize(pmap)
                    LLRmap[:, :, 0]=vpmap(DCpoly)
                    q1table[0, 0] = Q
                    alphatable[0, 0] = alpha
                    k1e = k1+1
                    k2e = k2+1
                    Lmax = L
    for index in range(1, c2):
        binHist=range(-2**11,2**11)
        coe = coeff[index]
        ic1 = int(cp.ceil(coe/8))
        ic2 = coe % 8
        if ic2 == 0:
            ic2 = 8

        A = cp.zeros((8,8))
        A[ic1-1, ic2-1] = 1
        B = idct(idct(A.T, norm='ortho').T, norm='ortho')
        AC = cp.rot90(fftconvolve(cp.rot90(Im, 2), cp.rot90(B, 2)), 2)  # This part is slow. Maybe look into cv2 replacement
        AC = AC[7:, 7:]
        ACpoly = AC[k1e-1::8, k2e-1::8]
        # choose shift for estimating unquantized distribution through
        # calibration
        if k1e < 5:
            k1cal = k1e + 1
        else:
            k1cal = k1e - 1
        if k2e < 5:
            k2cal = k2e + 1
        else:
            k2cal = k2e - 1
        ACcal = AC[k1cal-1::8, k2cal-1::8]
        binHist = cp.arange(-2**11, 2**11-1)+0.5
        binHist = cp.append(binHist, max(2**11, cp.max(ACcal)))
        binHist = cp.insert(binHist, 0, min(-2**11, cp.min(ACcal)))
        hcal = cp.histogram(ACcal, binHist)[0]
        hcalnorm = (hcal+1)/(cp.size(ACcal)+cp.size(binHist)-1)
        # estimate std dev of quantization error on DCT coeffs (quantization of
        # second compression plus rounding/truncation between first and second
        # compression)
        EAC = Edct[ic1-1::8, ic2-1::8]
        varE = cp.var(EAC)
        if index == 1:
            bias = cp.mean(EAC)
        else:
            bias = 0
        sig = cp.sqrt(qtable[ic1-1, ic2-1]**2 / 12 + varE)
        h0=cp.array(cp.take(hcalnorm,cp.round(cp.ndarray.flatten(ACpoly,order='F')).astype("int")+center))
        
        # estimate parameters of first compression
        [Q, alpha] = EMperiod(cp.ndarray.flatten(ACpoly,order='F'), minQ[ic1-1, ic2-1], maxQ[ic1-1, ic2-1], 0.95, h0, 5, 20, hcal, bias, sig)[:2]
        q1table[ic1-1, ic2-1] = Q
        alphatable[ic1-1, ic2-1] = alpha
        # simplified model
        nz = cp.count_nonzero(ACpoly)/cp.size(ACpoly)
        LLRmap_s[:, :, index] = LLR(ACpoly, nz, Q, int(cp.round(bias)), sig)
        # standard model
        ppu = cp.log(cp.divide(h1period(range(-2**11,2**11), Q, hcal, bias, sig),cp.take(hcalnorm,range(2**12))))
        ACpoly=cp.round(ACpoly).astype("int")+center
        LLRmap[:, :, index] = vpmap(ACpoly)
    OutputMap=correlate(cp.sum(LLRmap,2),cp.ones((3,3)),mode='reflect')
    return OutputMap



