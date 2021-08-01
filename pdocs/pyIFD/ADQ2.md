Module pyIFD.ADQ2
=================
This module provides the ADQ2 Algorithm

Functions
---------

    
`ceil2(x1)`
:   Applies ceil to vector x1, but if an element is close to an integer, it is raised by 0.5.
    
    Args:
        x1: Input vector
    
    Returns:
        x2: Output ceiling vector

    
`floor2(x1)`
:   Applies floor to vector x1, but if an element is close to an integer, it is lowered by 0.5.
    
    Args:
        x1: Input vector
    
    Returns:
        x2: Output floor vector

    
`getJmap(impath, ncomp=1, c1=1, c2=15)`
:   Main driver for ADQ2 algorithm.
    
    Args:
        impath: Input image path, required to be JPEG with extension .jpg
        ncomp: index of color component (1 = Y, 2 = Cb, 3 = Cr)
        c1: first DCT coefficient to consider (1 <= c1 <= 64)
        c2: last DCT coefficient to consider (1 <= c2 <= 64)
    
    Returns:
        maskTampered: estimated probability of being tampered for each 8x8 image block. Equivalent of OutputMap
        q1table: estimated quantization table of primary compression
        alphatable: mixture parameter for each DCT frequency
    
    Todos:
        * Check returns necessary

    
`ibdct(a, n=8)`
:   Performs an inverse discrete cosine transorm on array a with blocks of size nxn.
    
    Args:
        a: Array to be transformed. (2d array)
        n (optional, default=8): Size of blocks.
    
    Returns:
        b: Output after transform. (2d array)

    
`jpeg_rec(image)`
:   Simulate decompressed JPEG image from JPEG object.
    
    Args:
        image: JPEG object. (jpegio struct).
    
    Returns:
        IRecon: Reconstructed BGR image
        YCbCr: YCbCr image