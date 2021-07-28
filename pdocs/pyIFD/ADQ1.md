Module pyIFD.ADQ1
=================
This module provides the ADQ1 algorithm

Functions
---------

    
`ExtractYDCT(im)`
:   Determines YDCT.
    
    Args:
        im:
    
    Returns:
        YDCT:

    
`bdct(a, n=8)`
:   Computes bdct.
    
    Args:
        a:
        n (optional, default=8): 
    Returns:
        b:

    
`bdctmtx(n)`
:   Processes matrix using bdct.
    
    Args:
        n:
    
    Returns:
        m:

    
`detectDQ(impath)`
:   Main driver for ADQ1 algorithm
    
    Args:
        impath: Input image path
    
    Returns:
        OutputMap: Heatmap values for detected areas

    
`detectDQ_JPEG(im)`
:   Determines DQ for JPEG image.
    
    Args:
        im: Input image as read in by JPEGIO
    
    Returns:
        OutputMap: Heatmap values for detected areas

    
`detectDQ_NonJPEG(im)`
:   Determines DQ for non-JPEG.
    
    Args:
        im:
    
    Returns: 
        OutputMap: Heatmap values for detected areas

    
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

    
`vec2im(v, padsize=[0, 0], bsize=None, rows=None, cols=None)`
:   Converts vector to image.
    
    Args:
        v:
        padsize (optional, default=[0,0]):
        bsize (optional, default=None):
        rows (optional, default=None):
        cols (optional, default=None):
    
    Returns:
        im: