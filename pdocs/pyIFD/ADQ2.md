Module pyIFD.ADQ2
=================
This module provides the ADQ2 Algorithm

Functions
---------

    
`bdct(a, n=8)`
:   Applies bdct to block a of size nxn.
    
    Args:
        a:
        n (optional, default=8):
    
    Returns:
        b:

    
`bdctmtx(n)`
:   Generates bdct matrix of size nxn.
    
    Args:
        n:
    
    Returns:
        m:

    
`ceil2(x1)`
:   Applies ceil to vector x1, but if an element is close to an integer, it is raised by 0.5.
    
    Args:
        x1:
    
    Returns:
        x2:

    
`dequantize(qcoef, qtable)`
:   Dequantizes a coef table given a quant table.
    
    Args:
        qcoef:
        qtable:
    
    Returns:
        coef:

    
`floor2(x1)`
:   Applies floor to vector x1, but if an element is close to an integer, it is lowered by 0.5.
    
    Args:
        x1:
    
    Returns:
        x2:

    
`getJmap(impath, ncomp=1, c1=1, c2=15)`
:   Main driver for ADQ2 algorithm.
    
    Args:
        impath: Input image path, required to be JPEG with extension .jpg
    
    Returns:
        maskTampered:
        q1table:
        alphatable:
    
    Todos:
        * Check returns necessary
        * Check is maskTampered is equivalent to OutputMap (for naming convention)

    
`ibdct(a, n=8)`
:   Generates inverse bdct matrix of size nxn.
    
    Args:
        a:
        n (optional, default=8):
    
    Returns:
        b:

    
`im2vec(im, bsize, padsize=0)`
:   Converts image to vector.
    
    Args:
        im:
        bsize:
        padsize (optional, default=0):
    
    Returns:
        v:
        rows:
        cols:

    
`jpeg_rec(image)`
:   Simulate decompressed JPEG image from JPEG object.
    
    Args:
        image:
    
    Returns:
        I:
        YCbCr:

    
`vec2im(v, padsize=[0, 0], bsize=None, rows=None, cols=None)`
:   Converts vector to image.
    
    Args:
        v:
        padsize (optional, default)=[0,0]):
        bsize (optional, default=None):
        rows (optional, defeault=None):
        cols (optional, default=None):
    
    Returns:
        im