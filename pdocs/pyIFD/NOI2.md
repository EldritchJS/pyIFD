Module pyIFD.NOI2
=================
This module provides the NOI2 algorithm

Functions
---------

    
`GetNoiseMaps(impath, sizeThreshold=1760, filter_type='rand', filter_size=4, block_rad=8)`
:   Main driver for NOI2 algorithm.
    
    Args:
        impath:
        sizeThreshold (optional, default=55*25):
        filter_type (optional, default='rand'):
        filter_size (optional, default=4):
        block_rad (optional, default=8):
    
    Returns:
        estV: Equivalent to OutputMap

    
`GetNoiseMaps_hdd(im, filter_type, filter_size, block_rad)`
:   Outputs variance estimates for im. Equivalent to GetNoiseMaps_ram
    
    Args:
        im: Image to be processed.
        filter_type: Type of filter. Must be one of ('haar','dct','rand')
        filter_size: the size of the support of the filter
        block_rad: the size of the local blocks
    
    Returns:
        estV: estimated local noise variance
    TODO:
        * Consider removing the ram function path.

    
`GetNoiseMaps_ram(im, filter_type, filter_size, block_rad)`
:   Outputs variance estimates for im.
    
    Args:
        im: Image to be processed.
        filter_type: Type of filter. Must be one of ('haar','dct','rand')
        filter_size: the size of the support of the filter
        block_rad: the size of the local blocks
    
    Returns:
        estV: estimated local noise variance

    
`block_avg(X, d, pad='zero')`
:   Computes the avg of elements for all overlapping dxd windows in data X, where d = 2*rad+1.
    
    Args:
        X: an [nx,ny,ns] array as a stack of ns images of size [nx,ny]
        rad: radius of the sliding window, i.e., window size = (2*rad+1)*(2*rad+1)
        pad (optional, default='zero'): padding patterns
    
    Returns:
        Y: sum of elements for all overlapping dxd windows

    
`conv2(x, y, mode='same')`
:   Computes standard 2d convolution for matrices x and y.
    
    Args:
        x: 2d matrix.
        y: 2d matrix.
        mode (optional, default='same'):
    
    Returns:
        computation:
    
    Todos:
        * Sort out return

    
`dct2mtx(n, order)`
:   Generates matrices corresponding to 2D-DCT transform.
    
    Args:
        N: size of 2D-DCT basis (N x N)
        ord: order of the obtained DCT basis
    
    Returns:
        mtx: 3D matrices of dimension (NxNxN^2)

    
`haar2mtx(n)`
:   Generates haar filter of size (n,n,n**2).
    
    Args:
        n: Positive integer.
    
    Returns:
        mtx: nxn filter array.

    
`localNoiVarEstimate_hdd(noi, ft, fz, br)`
:   Computes local noise variance estimation using kurtosis.
    
    Args:
        noisyIm: input noisy image
        filter_type: the type of band-pass filter used supported types, "dct", "haar", "rand"
        filter_size: the size of the support of the filter
        block_rad: the size of the local blocks
    
    Returns:
        estVar: estimated local noise variance

    
`rnd2mtx(n)`
:   Generates matrices corresponding to random orthnormal transform.
    
    Args:
       N: size of 2D random basis (N x N)
    
    Returns:
       mtx: 3D matrices of dimension (NxNxN^2)