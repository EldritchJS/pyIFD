"""
This module provides the DCT algorithm

JPEG-block-artifact-based detector, solution 2 (leveraging Discrete Cosine Transforms).

Based on code from:
Ye, Shuiming, Qibin Sun, and Ee-Chien Chang. "Detecting digital image forgeries
by measuring inconsistencies of blocking artifact." In Multimedia and Expo, 2007
IEEE International Conference on, pp. 12-15. IEEE, 2007.
"""

import numpy as np
import jpegio as jio
from pyIFD.util import dequantize, extrema


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def hist3d(arr, bins):
    """
    """
    arrs = np.shape(arr)
    out = np.zeros((arrs[0], arrs[1], len(bins)))
    for x in range(arrs[0]):
        for y in range(arrs[1]):
            out[x, y, :-1] = np.histogram(arr[x, y, :], bins)[0]
            out[x, y, -1] = np.count_nonzero(arr[x, y, :] == bins[-1])
    return out


def DCT(impath):
    """
    Main driver for DCT algorithm.

    Args:
        impath: Input image path
    Returns:
        OutputMap: OutputMap
    """
    if impath[-4:] == ".jpg":
        try:
            OutputMap = GetDCTArtifact(jio.read(impath))
        except Exception as e:
            print('JPEGIO exception: ' + str(e))
            return
    else:
        print('Only .jpg supported')
    return OutputMap


def GetDCTArtifact(im):
    """
    Determines DCT artifacts.

    Args:
        im: Input image

    Returns:
        BMat: OutputMap
    """
    MaxCoeffs = 32
    coeff = [1, 9, 2, 3, 10, 17, 25, 18, 11, 4, 5, 12, 19, 26, 33, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 57, 50, 43, 36, 29, 22, 15, 8, 16,
             23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 24, 32, 39, 46, 53, 60, 61, 54, 47, 40, 48, 55, 62, 63, 56, 64]

    # Depending on whether im was created using jpeg_read (and thus is a struct)
    # or CleanUpImage(/imread), run a different code block for DCT block
    # extraction

    Q = im.quant_tables[0]
    YDCT = im.coef_arrays[0]
    YDCT = dequantize(YDCT, Q)
    imSize = np.shape(YDCT)
    YDCT_Block = np.reshape(YDCT, (8, round(imSize[0]/8), 8, round(imSize[1]/8)), order='F')
    YDCT_Block = np.transpose(YDCT_Block, [0, 2, 1, 3])
    YDCT_Block = np.reshape(YDCT_Block, (8, 8, round(imSize[0]*imSize[1]/64)), order='F')

    DCTHists = hist3d(YDCT_Block, np.arange(-257, 258))
    DCTHists = DCTHists[:, :, 1:-1]

    QEst = np.zeros((8, 8))

    # skip the DC term
    for coeffIndex in range(1, MaxCoeffs):
        NoPeaks = False
        coe = coeff[coeffIndex]
        startY = coe % 8
        if startY == 0:
            startY = 8
        startX = int(np.ceil(coe/8))
        DCTHist = np.ndarray.flatten(DCTHists[startY-1, startX-1, :], order='F')
        HistFFT = np.fft.fft(DCTHist)-1
        Power = abs(HistFFT)
        PowerFilterSize = 3
        g = matlab_style_gauss2D([1, 51], PowerFilterSize)
        PowerFilt = np.convolve(Power, np.ravel(g), 'same')
        Valley = 1
        while (PowerFilt[Valley-1] <= PowerFilt[Valley]):
            Valley += 1
        Valley += 1
        while (Valley < len(PowerFilt)-1) and (PowerFilt[Valley-1] >= PowerFilt[Valley]):
            Valley = Valley+1
        if Valley*2 < len(Power)*0.8:
            Power = Power[Valley-1:-Valley]
        else:
            NoPeaks = True
        Diff2 = np.diff(Power, 2)
        if len(Diff2) == 0:
            Diff2 = 0
        g = matlab_style_gauss2D([1, 51], 5)
        yfilt = np.convolve(Diff2, np.ravel(g), 'same')
        yfilt[yfilt > (min(yfilt)/5)] = 0
        imin = extrema(yfilt)
        if NoPeaks is True:
            imin = []
        QEst[startY-1, startX-1] = len(imin)
    D = np.tile(QEst[:, :, None], [1, 1, np.shape(YDCT_Block)[2]])
    with np.errstate(invalid='ignore', divide='ignore'):
        BMat = abs(YDCT_Block-np.round(YDCT_Block/D)*D)
    BMat[np.isnan(BMat)] = 0
    BMat = np.sum(np.sum(BMat, 0), 0)
    BMat = np.reshape(BMat, (int(imSize[0]/8), int(imSize[1]/8)), order='F')
    return BMat.astype("uint8")
