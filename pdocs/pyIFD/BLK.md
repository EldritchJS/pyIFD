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
        blockData:
    
    Returns:
        b:

    
`GetBlockGrid(impath)`
:   Main driver for BLK algorithm.
    
    Args:
        impath: Input image path
    
    Returns:
        b:
        eH:
        HorzMid:
        eV:
        VertMid:
        BlockDiff:
    
    Todos:
        * Check if all returns necessary
        * Check which, if any, corresponds to OutputMap

    
`GetBlockView(A, block=(8, 8))`
:   Splits A into blocks of size blocks.
    
    Args:
        A:
        block (optional, default=(8, 8)):
    
    Returns:
        ast(A, shape=shape, strides=strides):