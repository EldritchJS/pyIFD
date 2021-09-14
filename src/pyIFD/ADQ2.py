"""
This module provides the ADQ2 Algorithm

Aligned-double-JPEG-compression-based detector, solution 2.

Algorithm attribution:
T. Bianchi, A. De Rosa, and A. Piva, "IMPROVED DCT COEFFICIENT ANALYSIS
FOR FORGERY LOCALIZATION IN JPEG IMAGES", ICASSP 2011, Prague, Czech Republic,
2011, pp. 2444-2447.

Based on code from:
Zampoglou, M., Papadopoulos, S., & Kompatsiaris, Y. (2017). Large-scale evaluation of splicing localization algorithms for web images. Multimedia Tools and Applications, 76(4), 4801–4834.    
"""

import cupy as cp
from scipy.signal import medfilt2d
import jpegio as jio
import math
import cv2
from pyIFD.util import bdctmtx, im2vec, vec2im, dequantize, bdct


def ibdct(a, n=8):
    """
    Performs an inverse discrete cosine transorm on array a with blocks of size nxn.

    Args:
        a: Array to be transformed. (2d array)
        n (optional, default=8): Size of blocks.

    Returns:
        b: Output after transform. (2d array)
    """
    dctm = bdctmtx(n)

    [v, r, c] = im2vec(a, n)
    b = vec2im(dctm.T @ v, 0, n, r, c)
    return b


def jpeg_rec(image):
    """
    Simulate decompressed JPEG image from JPEG object.

    Args:
        image: JPEG object. (jpegio struct).

    Returns:
        IRecon: Reconstructed BGR image
        YCbCr: YCbCr image
    """

    Y = ibdct(dequantize(image.coef_arrays[0], image.quant_tables[0]))
    Y += 128

    if(image.image_components == 3):
        if(len(image.quant_tables) == 1):
            image.quant_tables[1] = image.quant_tables[0]
            image.quant_tables[2] = image.quant_tables[0]

        Cb = ibdct(dequantize(image.coef_arrays[1], image.quant_tables[1]))
        Cr = ibdct(dequantize(image.coef_arrays[2], image.quant_tables[1]))

        [r, c] = cp.shape(Y)
        [rC, cC] = cp.shape(Cb)

        if(math.ceil(r/rC) == 2) and (math.ceil(c/cC) == 2):  # 4:2:0
            kronMat = cp.ones((2, 2))
        elif(math.ceil(r/rC) == 1) and (math.ceil(c/cC) == 4):  # 4:1:1
            kronMat = cp.ones((1, 4))
        elif(math.ceil(r/rC) == 1) and (math.ceil(c/cC) == 2):  # 4:2:2
            kronMat = cp.ones((1, 4))
        elif(math.ceil(r/rC) == 1) and (math.ceil(c/cC) == 1):  # 4:4:4
            kronMat = cp.ones((1, 1))
        elif(math.ceil(r/rC) == 2) and (math.ceil(c/cC) == 1):  # 4:4:0
            kronMat = cp.ones((2, 1))
        else:
            raise Exception("Subsampling method not recognized: "+str(cp.shape(Y))+" "+str(cp.shape(Cr)))

        Cb = cp.kron(Cb, kronMat)+128
        Cr = cp.kron(Cr, kronMat)+128

        Cb = Cb[:r, :c]
        Cr = Cr[:r, :c]

        IRecon = cp.zeros((r, c, 3))
        IRecon[:, :, 0] = (Y+1.402*(Cr-128))
        IRecon[:, :, 1] = (Y-0.34414*(Cb-128)-0.71414*(Cr-128))
        IRecon[:, :, 2] = (Y+1.772*(Cb-128))
        YCbCr = cp.concatenate((Y, Cb, Cr), axis=1)
    else:
        IRecon = cp.tile(Y, [1, 1, 3])
        YCbCr = cv2.cvtColor(IRecon, cv2.COLOR_BGR2YCR_CB)

    return [IRecon, YCbCr]


def floor2(x1):
    """
    Applies floor to vector x1, but if an element is close to an integer, it is lowered by 0.5.

    Args:
        x1: Icput vector

    Returns:
        x2: Output floor vector
    """
    tol = 1e-12
    x2 = cp.floor(x1)
    idx = cp.where(cp.absolute(x1-x2) < tol)
    x2[idx] = x1[idx]-0.5
    return x2


def ceil2(x1):
    """
    Applies ceil to vector x1, but if an element is close to an integer, it is raised by 0.5.

    Args:
        x1: Icput vector

    Returns:
        x2: Output ceiling vector
    """
    tol = 1e-12
    x2 = cp.ceil(x1)
    idx = cp.where(cp.absolute(x1-x2) < tol)
    x2[idx] = x1[idx] + 0.5
    return x2


def getJmap(impath, ncomp=1, c1=1, c2=15):
    """
    Main driver for ADQ2 algorithm.

    Args:
        impath: Icput image path, required to be JPEG with extension .jpg
        ncomp: index of color component (1 = Y, 2 = Cb, 3 = Cr)
        c1: first DCT coefficient to consider (1 <= c1 <= 64)
        c2: last DCT coefficient to consider (1 <= c2 <= 64)

    Returns:
        maskTampered: estimated probability of being tampered for each 8x8 image block. Equivalent of OutputMap
        q1table: estimated quantization table of primary compression
        alphatable: mixture parameter for each DCT frequency

    Todos:
        * Check returns necessary
    """
    if impath[-4:] == ".jpg":
        try:
            image = jio.read(impath)
        except Exception as e:
            print('JPEGIO exception: ' + str(e))
            return
    else:
        print("Only .jpg accepted")
        return
    ncomp -= 1  # indexing
    coeffArray = image.coef_arrays[ncomp]
    qtable = image.quant_tables[image.comp_info[ncomp].quant_tbl_no]

    # estimate rounding and truncation error
    ImIn = jpeg_rec(image)[0]
    Iint = ImIn.copy()
    Iint[Iint < 0] = 0
    Iint[Iint > 255] = 255
    E = ImIn-cp.double(cp.uint8(Iint+0.5))
    Edct = bdct(0.299*E[:, :, 0]+0.587*E[:, :, 1]+0.114*E[:, :, 2])
    Edct2 = cp.reshape(Edct, (1, cp.size(Edct)), order='F').copy()
    varE = cp.var(Edct2)

    # simulate coefficients without DQ effect
    Y = ibdct(dequantize(coeffArray, qtable))
    coeffArrayS = bdct(Y[1:, 1:])

    sizeCA = cp.shape(coeffArray)
    sizeCAS = cp.shape(coeffArrayS)
    coeff = [1, 9, 2, 3, 10, 17, 25, 18, 11, 4, 5, 12, 19, 26, 33, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 57, 50, 43, 36, 29, 22, 15, 8,
             16, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 24, 32, 39, 46, 53, 60, 61, 54, 47, 40, 48, 55, 62, 63, 56, 64]
    coeffFreq = cp.zeros((int(cp.size(coeffArray)/64), 1))
    coeffSmooth = cp.zeros((int(cp.size(coeffArrayS)/64), 1))
    errFreq = cp.zeros((int(cp.size(Edct)/64), 1))

    bppm = 0.5*cp.ones((int(cp.size(coeffArray)/64), 1))
    bppmTampered = 0.5*cp.ones((int(cp.size(coeffArray)/64), 1))

    q1table = 100*cp.ones(cp.shape(qtable))
    alphatable = cp.ones(cp.shape(qtable))
    Q1up = cp.concatenate((20*cp.ones(10), 30*cp.ones(5), 40*cp.ones(6), 64*cp.ones(7), 80*cp.ones(8), 88*cp.ones(28)))

    rangeC = cp.arange(c1-1, c2)
    for index in rangeC:
        coe = coeff[index]
        # load DCT coefficients at position index
        k = 0
        start = coe % 8
        if(start == 0):
            start = 8
        rangeL = cp.arange(start-1, sizeCA[1], 8)
        rangeI = cp.arange(math.ceil(coe/8)-1, sizeCA[0], 8)
        for rl in rangeL:
            for i in rangeI:
                coeffFreq[k] = coeffArray[i, rl]
                errFreq[k] = Edct[i, rl]
                k += 1
        k = 0
        rangeL = cp.arange(start-1, sizeCAS[1], 8)
        rangeI = cp.arange(math.ceil(coe/8)-1, sizeCAS[0], 8)
        for rl in rangeL:
            for i in rangeI:
                coeffSmooth[k] = coeffArrayS[i, rl]
                k += 1

        # get histogram of DCT coefficients
        binHist = cp.arange(-2**11, 2**11-1)+0.5
        binHist = cp.append(binHist, max(2**11, coeffFreq.max()))
        binHist = cp.insert(binHist, 0, min(-2**11, coeffFreq.min()))
        num4Bin = cp.histogram(coeffFreq, binHist)[0]

        # get histogram of DCT coeffs w/o DQ effect (prior model for
        # uncompressed image
        Q2 = qtable[math.floor((coe-1) / 8), (coe-1) % 8]
        binHist = cp.arange(-2**11, 2**11-1)+0.5
        binHist *= Q2
        binHist = cp.append(binHist, max(Q2*(2**11), coeffSmooth.max()))
        binHist = cp.insert(binHist, 0, min(Q2*(-2**11), coeffSmooth.min()))
        hsmooth = cp.histogram(coeffSmooth, binHist)[0]

        # get estimate of rounding/truncation error
        biasE = cp.mean(errFreq)

        # kernel for histogram smoothing
        sig = math.sqrt(varE)/Q2
        f = math.ceil(6*sig)
        p = cp.arange(-f, f+1)
        g = cp.exp(-p**2/sig**2/2)
        g = g/sum(g)

        binHist = cp.arange(-2**11, 2**11)
        lidx = cp.invert([binHist[i] != 0 for i in range(len(binHist))])
        hweight = 0.5*cp.ones((1, 2**12))[0]
        E = float('inf')
        Etmp = cp.ones((1, 99))[0]*float('inf')
        alphaest = 1
        Q1est = 1
        biasest = 0

        if(index == 0):
            bias = biasE
        else:
            bias = 0
        # estimate Q-factor of first compression
        rangeQ = cp.arange(1, Q1up[index]+1)
        for Q1 in rangeQ:
            for b in [bias]:
                alpha = 1
                if(Q2 % Q1 == 0):
                    diff = cp.square(hweight * (hsmooth-num4Bin))
                else:
                    # nhist * hsmooth = prior model for doubly compressed coefficient
                    nhist = Q1/Q2*(floor2((Q2/Q1)*(binHist+b/Q2+0.5))-ceil2((Q2/Q1)*(binHist+b/Q2-0.5))+1)
                    nhist = cp.convolve(g, nhist)
                    nhist = nhist[f:-f]
                    a1 = cp.multiply(hweight, cp.multiply(nhist, hsmooth)-hsmooth)
                    a2 = cp.multiply(hweight, hsmooth-num4Bin)
                    # Exclude zero bin from fitting
                    la1 = cp.ma.masked_array(a1, lidx).filled(0)
                    la2 = cp.ma.masked_array(a2, lidx).filled(0)
                    alpha = (-(la1 @ la2.T))/(la1 @ la1.T)
                    alpha = min(alpha, 1)
                    diff = (hweight*(alpha*a1+a2))**2
                KLD = sum(cp.ma.masked_array(diff, lidx).filled(0))
                if KLD < E and alpha > 0.25:
                    E = KLD.copy()
                    Q1est = Q1.copy()
                    alphaest = alpha
                if KLD < Etmp[int(Q1) - 1]:
                    Etmp[int(Q1) - 1] = KLD
                    biasest = b
        Q1 = Q1est.copy()
        nhist = Q1 / Q2 * (floor2((Q2 / Q1) * (binHist + biasest / Q2 + 0.5)) - ceil2((Q2 / Q1) * (binHist + biasest / Q2 - 0.5)) + 1)
        nhist = cp.convolve(g, nhist)
        nhist = nhist[f:-f]
        nhist = alpha * nhist + 1 - alpha

        ppt = cp.mean(nhist) / (nhist + cp.mean(nhist))
        alpha = alphaest
        q1table[math.floor((coe - 1) / 8), (coe - 1) % 8] = Q1est
        alphatable[math.floor((coe - 1) / 8), (coe - 1) % 8] = alpha
        # compute probabilities if DQ effect is present
        if(Q2 % Q1est > 0):
            # index
            nhist = Q1est / Q2 * (floor2((Q2 / Q1est) * (binHist + biasest / Q2 + 0.5)) - ceil2((Q2 / Q1est) * (binHist + biasest / Q2 - 0.5)) + 1)
            # histogram smoothing (avoids false alarms)
            nhist = cp.convolve(g, nhist)
            nhist = nhist[f:-f]
            nhist = alpha * nhist + 1 - alpha
            ppu = nhist / (nhist + cp.mean(nhist))
            ppt = cp.mean(nhist) / (nhist + cp.mean(nhist))
            # set zeroed coefficients as non-informative
            ppu[2**11] = 0.5
            ppt[2**11] = 0.5
            idx = cp.floor(coeffFreq+2**11).astype(int)
            bppm = bppm * ppu[idx]
            bppmTampered = bppmTampered * ppt[idx]
    maskTampered = bppmTampered / (bppm + bppmTampered)
    maskTampered = cp.reshape(maskTampered, (int(sizeCA[0] / 8), int(sizeCA[1] / 8)), order='F')
    # apply median filter to highlight connected regions
    maskTampered = medfilt2d(maskTampered, [5, 5])
    return [maskTampered, q1table, alphatable]
