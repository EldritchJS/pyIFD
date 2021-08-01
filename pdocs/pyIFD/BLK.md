Module pyIFD.BLK
================
This module provides the BLK algorithm

Functions
---------

    
`ApplyFunction(M, blk_size=(8, 8))`
:   Applies BlockValue function to blocks of input
    
    Args:
        M:
        blk_size (optional, default=(8,8)):
    
    Returns:
        OutputMap:

    
`BlockValue(blockData)`
:   Get the per-block feature of blockData.
    
    Args:
        blockData: Input 2d array to extract features from.
    
    Returns:
        b: A float containing features of blockData

    
`GetBlockGrid(impath)`
:   Main driver for BLK algorithm.
    
    Args:
        impath: Input image path
    
    Returns:
        b: Main output of BLK. (2d array). This output corresponds to OutputMap
        eH:
        HorzMid:
        eV:
        VertMid:
        BlockDiff:
    
    Todos:
        * Check if all returns necessary

    
`GetBlockView(A, block=(8, 8))`
:   Splits A into blocks of size blocks.
    
    Args:
        A: 2d array A to be split up.
        block (optional, default=(8, 8)):
    
    Returns:
        ast(A, shape=shape, strides=strides): 4d array. First two dimensions give the coordinates of the block. Second two dimensions give the block data.