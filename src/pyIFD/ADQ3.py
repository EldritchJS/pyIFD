"""
This module provides the ADQ3 algorithm

Aligned-double-JPEG-compression-based detector, solution 3.

Algorithm attribution:
Amerini, Irene, Rudy Becarelli, Roberto Caldelli, and Andrea Del Mastio.
"Splicing forgeries localization through the use of first digit features."
In Information Forensics and Security (WIFS), 2014 IEEE International
Workshop on, pp. 143-148. IEEE, 2014.

Based on code from:
Zampoglou, M., Papadopoulos, S., & Kompatsiaris, Y. (2017). Large-scale evaluation of splicing localization algorithms for web images. Multimedia Tools and Applications, 76(4), 4801–4834.
"""

import cupy as cp
import jpegio as jio
import os
from pyIFD.util import dequantize

SupportVector = cp.load(os.path.join(os.path.dirname(__file__), 'SupportVector.cpy'), allow_pickle=True)
AlphaHat = cp.load(os.path.join(os.path.dirname(__file__), 'AlphaHat.cpy'), allow_pickle=True)
bias = cp.array([0.10431149, -0.25288239, -0.2689174, 0.39425104, -1.11269764, -1.15730589, -1.18658372, -0.9444815, -3.46445309, -2.9434976])


def BenfordDQ(impath):
    """
    Main driver for ADQ3 algorithm.

    Args:
        impath: Icput image path, required to be JPEG with extension .jpg

    Returns:
        OutputMap: Output of ADQ3 algorithm (2D array).
    """
    if impath[-4:] == '.jpg':
        try:
            im = jio.read(impath)
        except Exception as e:
            print('Exception in JPEGIO read: ' + str(e))
            return
    else:
        print("Only .jpg accepted.")
        return
    Quality = EstimateJPEGQuality(im)
    QualityInd = int(cp.round((Quality-50)/5+1))
    if QualityInd > 10:
        QualityInd = 10
    elif QualityInd < 1:
        QualityInd = 1
    c1 = 2
    c2 = 10
    ncomp = 1
    digitBinsToKeep = [2, 5, 7]
    block = im
    YCoef = im.coef_arrays[ncomp-1]
    Step = 8
    BlockSize = 64
    maxX = cp.shape(YCoef)[0]+1-BlockSize
    maxY = cp.shape(YCoef)[1]+1-BlockSize
    OutputMap = cp.zeros((int(cp.ceil(maxX-1)/Step+1), int(cp.ceil(maxY-1)/Step+1)))
    if cp.shape(im.coef_arrays[0])[0] < BlockSize:
        return 0
    for X in range(1, cp.shape(YCoef)[0]+1, Step):
        StartX = min(X, cp.shape(YCoef)[0]-BlockSize+1)
        for Y in range(1, cp.shape(YCoef)[1]+1, Step):
            StartY = min(Y, cp.shape(YCoef)[1]-BlockSize+1)
            block.coef_arrays[ncomp-1] = YCoef[StartX-1:StartX+BlockSize-1, StartY-1:StartY+BlockSize-1]
            Features = ExtractFeatures(block, c1, c2, ncomp, digitBinsToKeep)
            Features /= 64
            Dist = svmdecision(Features, QualityInd-1)
            OutputMap[int(cp.ceil((StartX-1)/Step)), int(cp.ceil((StartY-1)/Step))] = Dist
    OutputMap = cp.concatenate((cp.tile(OutputMap[0, :], (int(cp.ceil(BlockSize / 2 / Step)), 1)), OutputMap), axis=0)
    OutputMap = cp.concatenate((cp.tile(OutputMap[:, 0], (int(cp.ceil(BlockSize / 2 / Step)), 1)).T, OutputMap), axis=1)
    return OutputMap


def EstimateJPEGQuality(imIn):
    """
    Estimates the quality of JPEG object.

    Args:
        imIn: jpegio struct

    Returns:
        Quality: 0-100 integer
    """
    if(len(imIn.quant_tables) == 1):
        imIn.quant_tables[1] = imIn.quant_tables[0]
    YQuality = 100-(cp.sum(imIn.quant_tables[0])-imIn.quant_tables[0][0][0])/63
    CrCbQuality = 100-(cp.sum(imIn.quant_tables[1])-imIn.quant_tables[0][0][0])/63
    Diff = abs(YQuality-CrCbQuality)*0.98
    Quality = (YQuality+2*CrCbQuality)/3+Diff
    return Quality


def ExtractFeatures(im, c1, c2, ncomp, digitBinsToKeep):
    """
    This function extracts a descriptor feature based on the first-digit distribution of DCT coefficients of an image. It is needed by BenfordDQ.

     Args:
         c1: first DCT coefficient to be taken into account, DC term included
         c2: final DCT coefficient to be taken into account, DC term included
         ncomp: component from which to extract the feature (1 corresponds to the Y component)
         digitBinsToKeep: digits for which to keep their frequency

    Returns:
         output: Flattened feature vector
    """
    coeffArray = im.coef_arrays[ncomp-1]
    qtable = im.quant_tables[im.comp_info[ncomp].quant_tbl_no-1]
    Y = dequantize(coeffArray, qtable)
    coeff = [1, 9, 2, 3, 10, 17, 25, 18, 11, 4, 5, 12, 19, 26, 33, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 57, 50, 43, 36, 29, 22, 15, 8, 16,
             23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 24, 32, 39, 46, 53, 60, 61, 54, 47, 40, 48, 55, 62, 63, 56, 64]
    sizeCA = cp.shape(coeffArray)
    digitHist = cp.zeros((c2-c1+1, 10))
    for index in range(c1, c2+1):
        coe = coeff[index-1]
        start = coe % 8
        if start == 0:
            start = 8
        coeffFreq=cp.ndarray.flatten(Y[int(cp.ceil(coe/8))-1:sizeCA[0]-1:8, start-1:sizeCA[1]:8], order='F')
        NumOfDigits = (cp.floor(cp.log10(abs(coeffFreq) + 0.5)) + 1)
        tmp = [10**(i-1) for i in cp.array(NumOfDigits)]
        FirstDigit = cp.floor(cp.divide(abs(coeffFreq), tmp)).astype("uint8")
        binHist = list(cp.arange(0.5, 9.5, 1))
        binHist.insert(0, -float('Inf'))
        binHist.append(float('Inf'))
        digitHist[index-c1, :] = cp.histogram(FirstDigit, binHist)[0]
    HistToKeep = digitHist[:, digitBinsToKeep]
    return cp.ndarray.flatten(HistToKeep)


def svmdecision(Xnew, index):
    """
    Uses given index of svm to classify Xnew.

    Args:
        Xnew: Array to be classifed
        index: Index of SVM to use to classify

    Returns:
        f: 2d array of svm decision output.
    """
    f = cp.dot(cp.tanh(SupportVector[index] @ cp.transpose(Xnew)-1), AlphaHat[index]) + bias[index]
    return f