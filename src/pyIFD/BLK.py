"""
This module provides the BLK algorithm

JPEG-block-artifact-based detector, solution 1.

Algorithm attribution:
Li, Weihai, Yuan Yuan, and Nenghai Yu. "Passive detection of doctored JPEG
image via block artifact grid extraction." Signal Processing 89, no. 9 (2009):
1821-1829.

Based on code from:
Zampoglou, M., Papadopoulos, S., & Kompatsiaris, Y. (2017). Large-scale evaluation of splicing localization algorithms for web images. Multimedia Tools and Applications, 76(4), 4801–4834.
"""

import cupy as cp
from scipy.ndimage import median_filter
from scipy.ndimage import convolve
from PIL import Image
from cupy.lib.stride_tricks import as_strided as ast
import cv2


def BlockValue(blockData, blk_size):
    """
    Get the per-block feature of blockData.

    Args:
        blockData: Icput 2d array to extract features from.

    Returns:
        b: A float containing features of blockData
    """
    if cp.shape(blockData) != blk_size:
        blockData=cp.pad(blockData, ((0,8-cp.shape(blockData)[0]),(0,8-cp.shape(blockData)[1])), 'constant', constant_values=(1,1))
    Max1 = cp.max(cp.sum(blockData[1:7, 1:7], 0))  # Inner rows and columns added rowwise
    Min1 = cp.min(cp.sum(blockData[1:7, (0, 7)], 0))  # First and last columns, inner rows, added rowwise
    Max2 = cp.max(cp.sum(blockData[1:7, 1:7], 1))  # Inner rows and columns added columnwise
    Min2 = cp.min(cp.sum(blockData[(0, 7), 1:7], 1))  # First and last rows, inner colums, added columnwise

    b = Max1-Min1+Max2-Min2
    return b


def GetBlockView(A, block=(8, 8)):
    """
    Splits A into blocks of size blocks.

    Args:
        A: 2d array A to be split up.
        block (optional, default=(8, 8)):

    Returns:
        ast(A, shape=shape, strides=strides): 4d array. First two dimensions give the coordinates of the block. Second two dimensions give the block data.
    """
    shape = (int(cp.floor(A.shape[0] / block[0])), int(cp.floor(A.shape[1] / block[1]))) + block
    strides = (block[0]*A.strides[0], block[1]*A.strides[1]) + A.strides
    return ast(A, shape=shape, strides=strides)


def ApplyFunction(M, blk_size=(8, 8)):
    """
    Applies BlockValue function to blocks of icput

    Args:
        M:
        blk_size (optional, default=(8,8)):

    Returns:
        OutputMap:
    """
    Blocks=cp.ones((int(cp.ceil(cp.shape(M)[0]/blk_size[0])), int(cp.ceil(cp.shape(M)[1]/blk_size[1])), blk_size[0], blk_size[1]))
    Blocks[:int(cp.floor(cp.shape(M)[0]/blk_size[0])), :int(cp.floor(cp.shape(M)[1]/blk_size[1])), :, :] = GetBlockView(M, block=blk_size)
    OutputMap = cp.zeros(cp.shape(Blocks)[:2])
    for x in range(Blocks.shape[0]):
        for y in range(Blocks.shape[1]):
            OutputMap[x, y] = BlockValue(Blocks[x, y], blk_size)
    return OutputMap


def GetBlockGrid(impath):
    """
    Main driver for BLK algorithm.

    Args:
        impath: Icput image path

    Returns:
        b: Main output of BLK. (2d array). This output corresponds to OutputMap
        eH:
        HorzMid:
        eV:
        VertMid:
        BlockDiff:

    Todos:
        * Check if all returns necessary
    """
    im = cp.single(cv2.imread(impath))
    YCbCr = cp.double(cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB))
    Y = YCbCr[:, :, 0]

    # This thresh is used to remove extremely strong edges:
    # block edges are definitely going to be weak
    DiffThresh = 50

    # Accumulator size. Larger may overcome small splices, smaller may not
    # aggregate enough.
    AC = 33

    YH = cp.insert(Y, 0, Y[0, :], axis=0)
    YH = cp.append(YH, [Y[-1, :]], axis=0)
    Im2DiffY = -cp.diff(YH, 2, 0)
    Im2DiffY[cp.abs(Im2DiffY) > DiffThresh] = 0

    padsize = cp.round((AC-1)/2).astype(int)
    padded = cp.pad(Im2DiffY, ((0, 0), (padsize, padsize)), mode='symmetric')

    summedH = convolve(cp.abs(padded), cp.ones((1, AC)))
    summedH = summedH[:, padsize:-padsize]
    mid = median_filter(summedH, [AC, 1])
    eH = summedH-mid

    paddedHorz = cp.pad(eH, ((16, 16), (0, 0)), mode='symmetric')
    HMx = paddedHorz.shape[0]-32
    HMy = paddedHorz.shape[1]
    HorzMid = cp.zeros((HMx, HMy, 5))
    HorzMid[:, :, 0] = paddedHorz[0:HMx, :]
    HorzMid[:, :, 1] = paddedHorz[8:HMx+8, :]
    HorzMid[:, :, 2] = paddedHorz[16:HMx+16, :]
    HorzMid[:, :, 3] = paddedHorz[24:HMx+24, :]
    HorzMid[:, :, 4] = paddedHorz[32:HMx+32, :]

    HorzMid = cp.median(HorzMid, 2)

    YV = cp.insert(Y, 0, Y[:, 0], axis=1)
    YV = cp.insert(YV, -1, Y[:, -1], axis=1)
    Im2DiffX = -cp.diff(YV, 2, 1)
    Im2DiffX[cp.abs(Im2DiffX) > DiffThresh] = 0

    padded = cp.pad(Im2DiffX, ((padsize, padsize), (0, 0)), mode='symmetric')

    summedV = convolve(cp.abs(padded), cp.ones((AC, 1)))
    summedV = summedV[padsize:-padsize, :]

    mid = median_filter(summedV, [1, AC])
    eV = summedV-mid

    paddedVert = cp.pad(eV, ((0, 0), (padsize, padsize)), mode='symmetric')
    VMx = paddedVert.shape[0]
    VMy = paddedVert.shape[1]-32
    VertMid = cp.zeros((VMx, VMy, 5))

    VertMid[:, :, 0] = paddedVert[:, 0:VMy]
    VertMid[:, :, 1] = paddedVert[:, 8:VMy+8]
    VertMid[:, :, 2] = paddedVert[:, 16:VMy+16]
    VertMid[:, :, 3] = paddedVert[:, 24:VMy+24]
    VertMid[:, :, 4] = paddedVert[:, 32:VMy+32]
    VertMid = cp.median(VertMid, 2)

    BlockDiff = HorzMid+VertMid
    b = ApplyFunction(BlockDiff, (8, 8))

    return [b, eH, HorzMid, eV, VertMid, BlockDiff]
