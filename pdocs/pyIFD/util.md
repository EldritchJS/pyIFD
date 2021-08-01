Module pyIFD.util
=================
This file provides utility functions for the ADQ modules.

Functions
---------

    
`bdct(a, n=8)`
:   Performs dct on array via blocks of size nxn.
    
    Args:
        a: Array to perform dct on.
        n (optional, default=8): Size of blocks to perform dct on.
    Returns:
        b: Array after dct.

    
`bdctmtx(n)`
:   Produces bdct block matrix.
    
    Args:
        n: Size of block
    
    Returns:
        m: nxn array to performs dct with.

    
`dequantize(qcoef, qtable)`
:   Dequantizes a coef array given a quant table.
    Args:
        qcoef: Quantized coefficient array
        qtable: Table used to (de)quantize coef arrays. Must be the same size as qcoef.
    Returns:
        coef: Dequantized coef array. Same size as qcoef and qtable.

    
`extrema(x)`
:   Gets the local extrema points from a time series. This includes endpoints if necessary.
    Note that the indices will start counting from 1 to match MatLab.
    
    Args:
        x: time series vector
    
    Returns:
        imin: indices of XMIN

    
`im2vec(im, bsize, padsize=0)`
:   Converts image to vector.
    
    Args:
        im: Input image to be converted to a vector.
        bsize: Size of block of im to be converted to vec. Must be 1x2 non-negative int array.
        padsize (optional, default=0): Must be non-negative integers in a 1x2 array. Amount of zeros padded on each
    
    Returns:
        v: Output vector.
        rows: Number of rows of im after bsize and padsize are applied (before final flattening to vector).
        cols: Number of cols of im after bsize and padsize are applied (before final flattening to vector).

    
`vec2im(v, padsize=[0, 0], bsize=None, rows=None, cols=None)`
:   Converts vector to image.
    
    Args:
        v: input vector to be converted
        padsize (optional, default=[0,0]): Must be non-negative integers in a 1x2 array. Padsize dictates the amount of zeros padded for each of the two dimensions.
        bsize (optional, default=None): Block size. It's dimensions must multiply to the number of elements in v.
        rows (optional, default=None): Number of rows for output
        cols (optional, default=None): Number of cols for output
    
    Returns:
        im: Output image (2d numpy array)