"""
This file provides utility functions for the ADQ modules.
"""

import numpy as np
import math
from scipy import signal


def vec2im(v, padsize=[0, 0], bsize=None, rows=None, cols=None):
    """
    Converts vector to image.

    Args:
        v: input vector to be converted
        padsize (optional, default=[0,0]): Must be non-negative integers in a 1x2 array. Padsize dictates the amount of zeros padded for each of the two dimensions.
        bsize (optional, default=None): Block size. It's dimensions must multiply to the number of elements in v.
        rows (optional, default=None): Number of rows for output
        cols (optional, default=None): Number of cols for output

    Returns:
        im: Output image (2d numpy array)
    """
    [m, n] = np.shape(v)

    padsize = padsize+np.zeros((1, 2), dtype=int)[0]
    if(padsize.any() < 0):
        raise Exception("Pad size must not be negative")
    if bsize is None:
        bsize = math.floor(math.sqrt(m))
    bsize = bsize+np.zeros((1, 2), dtype=int)[0]

    if(np.prod(bsize) != m):
        raise Exception("Block size does not match size of input vectors.")

    if rows is None:
        rows = math.floor(math.sqrt(n))
    if cols is None:
        cols = math.ceil(n/rows)

    # make image
    y = bsize[0]+padsize[0]
    x = bsize[1]+padsize[1]
    t = np.zeros((y, x, rows*cols))
    t[:bsize[0], :bsize[1], :n] = np.reshape(v, (bsize[0], bsize[1], n), order='F')
    t = np.reshape(t, (y, x, rows, cols), order='F')
    t = np.reshape(np.transpose(t, [0, 2, 1, 3]), (y*rows, x*cols), order='F')
    im = t[:y*rows-padsize[0], :x*cols-padsize[1]]
    return im


def im2vec(im, bsize, padsize=0):
    """
    Converts image to vector.

    Args:
        im: Input image to be converted to a vector.
        bsize: Size of block of im to be converted to vec. Must be 1x2 non-negative int array.
        padsize (optional, default=0): Must be non-negative integers in a 1x2 array. Amount of zeros padded on each

    Returns:
        v: Output vector.
        rows: Number of rows of im after bsize and padsize are applied (before final flattening to vector).
        cols: Number of cols of im after bsize and padsize are applied (before final flattening to vector).
    """
    bsize = bsize+np.zeros((1, 2), dtype=int)[0]
    padsize = padsize+np.zeros((1, 2), dtype=int)[0]
    if(padsize.any() < 0):
        raise Exception("Pad size must not be negative")
    imsize = np.shape(im)
    y = bsize[0]+padsize[0]
    x = bsize[1]+padsize[1]
    rows = math.floor((imsize[0]+padsize[0])/y)
    cols = math.floor((imsize[1]+padsize[1])/x)
    t = np.zeros((y*rows, x*cols))
    imy = y*rows-padsize[0]
    imx = x*cols-padsize[1]
    t[:imy, :imx] = im[:imy, :imx]
    t = np.reshape(t, (y, rows, x, cols), order='F')
    t = np.reshape(np.transpose(t, [0, 2, 1, 3]), (y, x, rows*cols), order='F')
    v = t[:bsize[0], :bsize[1], :rows*cols]
    v = np.reshape(v, (y*x, rows*cols), order='F')
    return [v, rows, cols]


def bdctmtx(n):
    """
    Produces bdct block matrix.

    Args:
        n: Size of block

    Returns:
        m: nxn array to performs dct with.
    """
    [c, r] = np.meshgrid(range(8), range(8))
    [c0, r0] = np.meshgrid(r, r)
    [c1, r1] = np.meshgrid(c, c)
    x = np.zeros(np.shape(c))
    for i in range(n):
        for j in range(n):
            x[i, j] = math.sqrt(2/n)*math.cos(math.pi*(2*c[i, j]+1)*r[i, j]/(2*n))
    x[0, :] = x[0, :]/math.sqrt(2)
    x = x.flatten('F')
    m = np.zeros(np.shape(r0))
    for i in range(n**2):
        for j in range(n**2):
            m[i, j] = x[r0[i, j]+c0[i, j]*n]*x[r1[i, j]+c1[i, j]*n]
    return m


def bdct(a, n=8):
    """
    Performs dct on array via blocks of size nxn.

    Args:
        a: Array to perform dct on.
        n (optional, default=8): Size of blocks to perform dct on.
    Returns:
        b: Array after dct.
    """
    dctm = bdctmtx(n)

    [v, r, c] = im2vec(a, n)
    b = vec2im(dctm @ v, 0, n, r, c)
    return b


def dequantize(qcoef, qtable):
    """
    Dequantizes a coef array given a quant table.
    Args:
        qcoef: Quantized coefficient array
        qtable: Table used to (de)quantize coef arrays. Must be the same size as qcoef.
    Returns:
        coef: Dequantized coef array. Same size as qcoef and qtable.
    """
    blksz = np.shape(qtable)
    [v, r, c] = im2vec(qcoef, blksz)

    flat = np.array(qtable).flatten('F')
    vec = v*np.tile(flat, (np.shape(v)[1], 1)).T

    coef = vec2im(vec, 0, blksz, r, c)
    return coef


def extrema(x):
    """
    Gets the local extrema points from a time series. This includes endpoints if necessary.
    Note that the indices will start counting from 1 to match MatLab.

    Args:
        x: time series vector

    Returns:
        imin: indices of XMIN
    """
    x = np.asarray(x)
    imin = signal.argrelextrema(x, np.less)[0]
    if(x[-1] < x[-2]):  # Check last point
        imin = np.append(imin, len(x)-1)
    if(x[0] < x[1]):  # Check first point
        imin = np.insert(imin, 0, 0)
    xmin = x[imin]
    minorder = np.argsort(xmin)
    imin = imin[minorder]
    return imin+1
