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