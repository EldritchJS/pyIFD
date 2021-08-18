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
import numpy as np
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
        impath: Input image path
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

    t = np.reshape(t,(8,8),order='F').T
    t = np.floor((t * quality + 50)/100)
    t[t < 1] = 1

    t[t > 32767] = 32767  # max quantizer needed for 12 bits
    if (force_baseline):
        t[t > 255] = 255

    return t


def LLR(x, nz, Q, phase, sig):
    binHist=range(-2**11, 2**11)
    center=2**11
    # Finished review
    w = int(np.ceil(3*sig))
    k = list(range(-w,w+1))
    g = np.array([math.exp(-kk**2/sig**2/2) for kk in k])
    g = g/np.sum(g)
    N = np.size(x) / np.size(binHist)

    bppm = np.zeros(np.shape(binHist))
    bppm[center + phase::Q] = Q
    bppm[center + phase::-Q] = Q
    bppm = np.convolve(g, bppm)
    bppm = bppm[w:-w]
    bppm = (bppm*N + 1)
    LLRmap = np.log(bppm / np.mean(bppm))
    LLRmap[center] = nz * LLRmap[center]
    x=np.round(x).astype("int")+center
    def lmap(xx):
        return LLRmap[xx]
    vlmap=np.vectorize(lmap)
    L = vlmap(x)
    return L


def EMperiod(x, Qmin, Qmax, alpha0, h0, dLmin, maxIter, hcal, bias, sig):
    # Finished review
    Qvec = list(range(int(Qmin),int(Qmax)+1))
    alphavec = alpha0*np.ones(np.shape(Qvec))
    h1mat = np.zeros((len(Qvec), len(x)))
    for k in range(len(Qvec)):
        h1mat[k,:] = h1period(x, Qvec[k], hcal, bias, sig)
    Lvec = np.ones(np.shape(Qvec))*float('-inf')
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
            alphavec[k] = np.mean(beta0)
            # compute true log-likelihood of mixture
            L = np.sum(np.log(alphavec[k]*h0 + (1-alphavec[k])*h1mat[k,:]))
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
    N = np.sum(hcal)
    # simulate quantization
    if Q % 2 == 0:
        hs = np.ones(Q-1)
        hs=np.append(hs,0.5)
        hs=np.insert(hs,0, 0.5)
        ws = int(Q/2)
    else:
        hs = np.ones(Q)
        ws = int((Q-1)/2)
    h2 = np.convolve(hcal,hs)
    # simulate dequantization
    h1 = np.zeros(np.shape(binHist))
    h1[center::Q] = h2[center + ws:-ws:Q]
    h1[center::-Q] = h2[center + ws:ws-1:-Q]
    # simulate rounding/truncation
    w = int(np.ceil(3*sig))
    k = range(-w,w+1)
    g = [math.exp(-(kk+bias)**2/sig**2/2) for kk in k]
    h1 = np.convolve(h1, g)
    h1 = h1[w:-w]
    # normalize probability and use Laplace correction to avoid p1 = 0
    h1 /= sum(h1)
    h1 = (h1*N+1)/(N+np.size(binHist))
    x=np.array(x)
    p1=np.take(h1,np.round(np.nextafter(x,x+1)).astype("int")+center)
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
    q1table = np.ones((8,8))
    minQ = np.maximum(2,np.floor(qtable/np.sqrt(3)))
    maxQ = np.maximum(jpeg_qtable(50),qtable)
    # estimate rounding and truncation error
    Im = jpeg_rec(image)[0]
    ImTmp = Im.copy()
    ImTmp=np.maximum(0,ImTmp)
    ImTmp[ImTmp > 255] = 255
    E = Im - np.round(ImTmp)
    Edct = bdct(0.299 * E[:, :, 0] + 0.587 * E[:, :, 1] + 0.114 * E[:, :, 2])

    # compute DCT coeffs of decompressed image
    Im = ibdct(dequantize(coeffArray, qtable))
    coeff = [1, 9, 2, 3, 10, 17, 25, 18, 11, 4, 5, 12, 19, 26, 33, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 57, 50, 43, 36, 29, 22, 15, 8, 16, 23, 30, 37, 44, 51, 58, 59, 52,
             45, 38, 31, 24, 32, 39, 46, 53, 60, 61, 54, 47, 40, 48, 55, 62, 63, 56, 64]
    center = 2**11

    B = np.ones((8,8))/8
    DC = np.rot90(convolve2d(np.rot90(Im, 2), np.rot90(B, 2)), 2)
    DC = DC[7:, 7:]
    EDC = Edct[::8, ::8]
    varE = np.var(EDC)
    bias = np.mean(EDC)
    sig = np.sqrt(qtable[0, 0]**2 / 12 + varE)
    alphatable = np.ones((8,8))
    Ims=np.shape(Im)
    LLRmap = np.zeros((int(Ims[0]/8), int(Ims[1]/8), c2))
    LLRmap_s = np.zeros((int(Ims[0]/8), int(Ims[1]/8), c2))
    k1e = 1
    k2e = 1
    Lmax = -np.inf
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
                binHist = np.arange(-2**11, 2**11-1)+0.5
                binHist = np.append(binHist, max(2**11, np.max(DCcal)))
                binHist = np.insert(binHist, 0, min(-2**11, np.min(DCcal)))
                hcal = np.histogram(DCcal, binHist)[0]
                hcalnorm = (hcal+1)/(np.size(DCcal)+np.size(binHist)-1)
                # define mixture components
                h0=np.array(np.take(hcalnorm,np.round(np.ndarray.flatten(DCpoly,order='F')).astype("int")+center))
                # estimate parameters of first compression
                [Q, alpha, L] = EMperiod(np.ndarray.flatten(DCpoly,order='F'), minQ[0, 0], maxQ[0, 0], 0.95, h0, 5, 20, hcal, bias, sig)
                if L > Lmax:
                    # simplified model
                    nz = np.count_nonzero(DCpoly)/np.size(DCpoly)
                    LLRmap_s[:, :, 0] = LLR(DCpoly, nz, Q, int(np.round(bias)), sig)
                    # standard model
                    ppu = np.log(np.divide(h1period(range(-2**11,2**11), Q, hcal, bias, sig),np.take(hcalnorm,range(2**12))))
                    DCpoly=np.round(DCpoly).astype("int")+center
                    def pmap(xx):
                        return ppu[xx]
                    vpmap=np.vectorize(pmap)
                    LLRmap[:, :, 0]=vpmap(DCpoly)
                    q1table[0, 0] = Q
                    alphatable[0, 0] = alpha
                    k1e = k1+1
                    k2e = k2+1
                    Lmax = L
    for index in range(1, c2):
        binHist=range(-2**11,2**11)
        coe = coeff[index]
        ic1 = int(np.ceil(coe/8))
        ic2 = coe % 8
        if ic2 == 0:
            ic2 = 8

        A = np.zeros((8,8))
        A[ic1-1, ic2-1] = 1
        B = idct(idct(A.T, norm='ortho').T, norm='ortho')
        AC = np.rot90(fftconvolve(np.rot90(Im, 2), np.rot90(B, 2)), 2)  # This part is slow. Maybe look into cv2 replacement
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
        binHist = np.arange(-2**11, 2**11-1)+0.5
        binHist = np.append(binHist, max(2**11, np.max(ACcal)))
        binHist = np.insert(binHist, 0, min(-2**11, np.min(ACcal)))
        hcal = np.histogram(ACcal, binHist)[0]
        hcalnorm = (hcal+1)/(np.size(ACcal)+np.size(binHist)-1)
        # estimate std dev of quantization error on DCT coeffs (quantization of
        # second compression plus rounding/truncation between first and second
        # compression)
        EAC = Edct[ic1-1::8, ic2-1::8]
        varE = np.var(EAC)
        if index == 1:
            bias = np.mean(EAC)
        else:
            bias = 0
        sig = np.sqrt(qtable[ic1-1, ic2-1]**2 / 12 + varE)
        h0=np.array(np.take(hcalnorm,np.round(np.ndarray.flatten(ACpoly,order='F')).astype("int")+center))
        
        # estimate parameters of first compression
        [Q, alpha] = EMperiod(np.ndarray.flatten(ACpoly,order='F'), minQ[ic1-1, ic2-1], maxQ[ic1-1, ic2-1], 0.95, h0, 5, 20, hcal, bias, sig)[:2]
        q1table[ic1-1, ic2-1] = Q
        alphatable[ic1-1, ic2-1] = alpha
        # simplified model
        nz = np.count_nonzero(ACpoly)/np.size(ACpoly)
        LLRmap_s[:, :, index] = LLR(ACpoly, nz, Q, int(np.round(bias)), sig)
        # standard model
        ppu = np.log(np.divide(h1period(range(-2**11,2**11), Q, hcal, bias, sig),np.take(hcalnorm,range(2**12))))
        ACpoly=np.round(ACpoly).astype("int")+center
        LLRmap[:, :, index] = vpmap(ACpoly)
    OutputMap=correlate(np.sum(LLRmap,2),np.ones((3,3)),mode='reflect')
    return OutputMap



