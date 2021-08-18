"""
This module provides the CFA2 algorithm

Color-filter-array-artifact-based detector, solution 2.

Algorithm attribution:
Dirik, Ahmet Emir, and Nasir D. Memon. "Image tamper detection based on
demosaicing artifacts." In ICIP, pp. 1497-1500. 2009.

Based on code from:
Zampoglou, M., Papadopoulos, S., & Kompatsiaris, Y. (2017). Large-scale evaluation of splicing localization algorithms for web images. Multimedia Tools and Applications, 76(4), 4801–4834.
"""

import numpy as np
from scipy.ndimage import correlate
from numpy.lib.stride_tricks import as_strided as ast
import cv2

def bilinInterp(CFAIm, BinFilter, CFA):  # Possible this is provided in skimage or similar
    """
    Bilinear interpolation

    Args:
        CFAIm:
        BinFilter:
        CFA:
    Returns:
        OutputMap: Out_Im_Int
    """
    MaskMin = np.array([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]])
    MaskMaj = np.array([[0.0, 0.25, 0.0], [0.25, 1.0, 0.25], [0.0, 0.25, 0.0]])

    dCFA = np.diff(CFA, axis=0)
    dCFAT = np.diff(np.transpose(CFA), axis=0)

    if (np.argwhere(dCFA == 0).size > 0) or (np.argwhere(dCFAT == 0).size > 0):
        MaskMaj = MaskMaj*2
    Mask = np.tile(MaskMin[:, :, None], (1, 1, 3))
    Maj = np.argmax(np.sum(np.sum(BinFilter, 0), 0))
    Mask[:, :, Maj] = MaskMaj
    Out_Im = np.zeros(np.shape(CFAIm))

    for ii in range(3):
        Mixed_im = np.zeros((np.shape(CFAIm)[0], np.shape(CFAIm)[1]))
        Orig_Layer = CFAIm[:, :, ii]
        Interp_Layer = correlate(Orig_Layer, Mask[:, :, ii], mode='constant')  # imfilter(Orig_Layer,Mask[:, :, ii])
        Mixed_im[BinFilter[:, :, ii] == 0] = Interp_Layer[BinFilter[:, :, ii] == 0]
        Mixed_im[BinFilter[:, :, ii] == 1] = Orig_Layer[BinFilter[:, :, ii] == 1]
        Out_Im[:, :, ii] = Mixed_im

    Out_Im_Int = np.round(np.nextafter(Out_Im, Out_Im+1)).astype(int)

    return Out_Im_Int


def eval_block(block):   # Just more blockproc? Can this go in util?
    """
    Evaluated block.

    Args:
        impath: block_struc
    Returns:
        OutputMap: Out
    """
    Out = np.zeros(6)
    Out[0] = np.mean((np.double(block[:, :, 0]) - np.double(block[:, :, 3]))**2)
    Out[1] = np.mean((np.double(block[:, :, 1]) - np.double(block[:, :, 4]))**2)
    Out[2] = np.mean((np.double(block[:, :, 2]) - np.double(block[:, :, 5]))**2)

    Out[3] = np.std(np.ndarray.flatten(block[:, :, 0], order='F'), ddof=1)
    Out[4] = np.std(np.ndarray.flatten(block[:, :, 1], order='F'), ddof=1)
    Out[5] = np.std(np.ndarray.flatten(block[:, :, 2], order='F'), ddof=1)
    return Out


def GetBlockView(A, block=(16, 16)):
    """
    Splits A into blocks of size blocks.

    Args:
        A: 2d array A to be split up.
        block (optional, default=(8, 8)):

    Returns:
        ast(A, shape=shape, strides=strides): 4d array. First two dimensions give the coordinates of the block. Second two dimensions give the block data.
    """
    shape = (int(np.floor(A.shape[0] / block[0])), int(np.floor(A.shape[1] / block[1]))) + block
    strides = (block[0]*A.strides[0], block[1]*A.strides[1]) + A.strides
    return ast(A, shape=shape, strides=strides)


def ApplyFunction(M, blk_size=(16, 16)):
    """
    Applies BlockValue function to blocks of input

    Args:
        M: 3d array.
        blk_size (optional, default=(8,8)):

    Returns:
        OutputMap:
    """
    Blocks = np.zeros((int(np.floor(np.shape(M)[0]/blk_size[0])), int(np.floor(np.shape(M)[1]/blk_size[1])), blk_size[0], blk_size[1], 6))
    edges = np.mod(np.shape(M)[:2], blk_size)
    for i in range(6):
        Blocks[:, :, :, :, i] = GetBlockView(M[:, :, i])
    OutputMap = np.zeros((int(np.ceil(np.shape(M)[0]/blk_size[0])), int(np.ceil(np.shape(M)[1]/blk_size[1])), 6))
    for x in range(Blocks.shape[0]):
        for y in range(Blocks.shape[1]):
            OutputMap[x, y, :] = eval_block(Blocks[x, y])
    if edges[0] != 0:
        for y in range(Blocks.shape[1]):
            OutputMap[-1, y, :] = eval_block(M[-edges[0]:, y*blk_size[1]:(y+1)*blk_size[1], :])
    if edges[1] != 0:
        for x in range(Blocks.shape[0]):
            OutputMap[x, -1, :] = eval_block(M[x*blk_size[0]:(x+1)*blk_size[0]:, -edges[1]:, :])
    if edges[0] != 0 and edges[1] != 0:
        OutputMap[-1, -1, :] = eval_block(M[-edges[0]:, -edges[1]:, :])
    return OutputMap


def CFATamperDetection_F1(im):
    StdThresh = 5
    Depth = 3

    im = im[:np.round(np.floor(np.shape(im)[0]/(2**Depth))*(2**Depth)).astype(np.uint), :np.round(np.floor(np.shape(im)[1]/(2**Depth))*(2**Depth)).astype(np.uint), :]
    CFAList = np.array([[[2, 1], [3, 2]], [[2, 3], [1, 2]], [[3, 2], [2, 1]], [[1, 2], [2, 3]]])

    W1 = 16

    if np.shape(im)[0] < W1 or np.shape(im)[1] < W1:
        F1Map = np.zeros((np.shape(im)[0], np.shape(im)[1]))
        return F1Map
    MeanError = np.ones(np.size(CFAList))*np.inf
    Diffs = np.zeros((np.shape(CFAList)[0], int(np.ceil(np.shape(im)[0]/W1)*np.ceil(np.shape(im)[1]/W1))))
    F1Maps = np.zeros((np.shape(CFAList)[0], int(np.ceil(np.shape(im)[0]/W1)), int(np.ceil(np.shape(im)[1]/W1))))
    for TestArray in range(np.shape(CFAList)[0]):
        BinFilter = np.zeros((np.shape(im)[0], np.shape(im)[1], 3))
        ProcIm = np.zeros((np.shape(im)[0], np.shape(im)[1], 6))
        CFA = CFAList[TestArray]
        R = CFA == 1
        G = CFA == 2
        B = CFA == 3
        BinFilter[:, :, 0] = np.tile(R, (int(np.shape(im)[0]/2), int(np.shape(im)[1]/2)))
        BinFilter[:, :, 1] = np.tile(G, (int(np.shape(im)[0]/2), int(np.shape(im)[1]/2)))
        BinFilter[:, :, 2] = np.tile(B, (int(np.shape(im)[0]/2), int(np.shape(im)[1]/2)))
        CFAIm = im*BinFilter
        BilinIm = bilinInterp(CFAIm, BinFilter, CFA)
        ProcIm[:, :, 0:3] = im
        ProcIm[:, :, 3:6] = np.double(BilinIm)

        ProcIm = np.double(ProcIm)

        # BlockResult = blockproc(ProcIm, [W1 W1], @eval_block)
        BlockResult = ApplyFunction(ProcIm, (W1, W1))

        Stds = BlockResult[:, :, 3:6]
        BlockDiffs = BlockResult[:, :, :3]
        NonSmooth = Stds > StdThresh

        MeanError[TestArray] = np.mean(BlockDiffs[NonSmooth])
        with np.errstate(invalid='ignore'):
            BlockDiffs /= np.tile(np.sum(BlockDiffs, 2)[:, :, None], (1, 1, 3))

        Diffs[TestArray, :] = np.ndarray.flatten(BlockDiffs[:, :, 1], order='F')
        F1Maps[TestArray, :, :] = BlockDiffs[:, :, 1]
    Diffs[np.isnan(Diffs)] = 0
    val = int(np.argmin(MeanError))
    F1Map = F1Maps[val, :, :]
    F1Map[np.isnan(F1Map)] = 0
    CFAOut = CFAList[val] == 2
    return [F1Map,CFAOut]

def CFA2(impath):
    bgr = cv2.imread(impath)
    ImageIn = np.double(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    toCrop = np.mod(np.shape(ImageIn), 2)
    if toCrop[0] != 0:
        ImageIn = ImageIn[:-toCrop[0],:,:]
    if toCrop[1] != 0:
        ImageIn = ImageIn[:,:-toCrop[1],:]
    OutputMap = CFATamperDetection_F1(ImageIn)[0]
    return OutputMap
