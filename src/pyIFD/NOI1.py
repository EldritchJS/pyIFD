"""
This module provides the NOI1 algorithm

Noise-variance-inconsistency detector, solution 1.

Based on code from:
Mahdian, Babak, and Stanislav Saic. "Using noise inconsistencies for blind
image forensics." Image and Vision Computing 27, no. 10 (2009): 1497-1503.
"""

import numpy as np
from skimage.color import rgb2ycbcr
from PIL import Image
from pywt import dwt2


def GetNoiseMap(impath, BlockSize=8):
    """
    Main driver for NOI1 algorithm.

    Args:
        impath: Path to the image to be processed.
        BlockSize: the block size for noise variance estimation. Too small reduces quality, too large reduces localization accuracy

    Returns:
        OutputMap:
    """
    im = Image.open(impath)
    YCbCr = np.double(rgb2ycbcr(im))
    Y = np.round(YCbCr[:, :, 0])

    (cA1, (cH, cV, cD)) = dwt2(Y, 'db8')  # 2d discrete wavelet transform

    cD = cD[:int(np.floor(np.size(cD, 0)/BlockSize)*BlockSize), :int(np.floor(np.size(cD, 1)/BlockSize)*BlockSize)]

    Block = np.zeros((int(np.floor(np.size(cD, 0)/BlockSize)), int(np.floor(np.size(cD, 1)/BlockSize)), BlockSize**2))

    for ii in range(0, np.size(cD, 0)-1, BlockSize):
        for jj in range(0, np.size(cD, 1)-1, BlockSize):
            blockElements = cD[ii:ii+BlockSize, jj:jj+BlockSize]
            Block[int(ii/BlockSize), int(jj/BlockSize), :] = np.reshape(blockElements, (1, 1, np.size(blockElements)))

    OutputMap = np.median(np.abs(Block), 2)/0.6745

    return OutputMap
