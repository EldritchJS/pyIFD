"""
This module provides the NOI1 algorithm

Noise-variance-inconsistency detector, solution 1.

Algorithm attribution:
Mahdian, Babak, and Stanislav Saic. "Using noise inconsistencies for blind
image forensics." Image and Vision Computing 27, no. 10 (2009): 1497-1503.

Based on code from:
Zampoglou, M., Papadopoulos, S., & Kompatsiaris, Y. (2017). Large-scale evaluation of splicing localization algorithms for web images. Multimedia Tools and Applications, 76(4), 4801–4834.
"""

import cupy as cp
from skimage.color import rgb2ycbcr
from PIL import Image
from pywt import dwt2
import cv2


def GetNoiseMap(impath, BlockSize=8):
    """
    Main driver for NOI1 algorithm.

    Args:
        impath: Path to the image to be processed.
        BlockSize: the block size for noise variance estimation. Too small reduces quality, too large reduces localization accuracy

    Returns:
        OutputMap:
    """
    im = cv2.imread(impath)
    YCbCr = cp.double(cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB))
    Y = cp.round(YCbCr[:, :, 0])

    (cA1, (cH, cV, cD)) = dwt2(Y, 'db8')  # 2d discrete wavelet transform

    cD = cD[:int(cp.floor(cp.size(cD, 0)/BlockSize)*BlockSize), :int(cp.floor(cp.size(cD, 1)/BlockSize)*BlockSize)]

    Block = cp.zeros((int(cp.floor(cp.size(cD, 0)/BlockSize)), int(cp.floor(cp.size(cD, 1)/BlockSize)), BlockSize**2))

    for ii in range(0, cp.size(cD, 0)-1, BlockSize):
        for jj in range(0, cp.size(cD, 1)-1, BlockSize):
            blockElements = cD[ii:ii+BlockSize, jj:jj+BlockSize]
            Block[int(ii/BlockSize), int(jj/BlockSize), :] = cp.reshape(blockElements, (1, 1, cp.size(blockElements)))

    OutputMap = cp.median(cp.abs(Block), 2)/0.6745

    return OutputMap
