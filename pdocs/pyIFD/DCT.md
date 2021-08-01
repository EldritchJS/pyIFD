Module pyIFD.DCT
================
This module provides the DCT algorithm

Functions
---------

    
`DCT(impath)`
:   Main driver for DCT algorithm
    
    Args:
        impath: Input image path
    Returns:
        OutputMap: OutputMap

    
`GetDCTArtifact(im)`
:   Determines DCT artifacts.
    
    Args:
        im: Input image
    
    Returns:
        BMat: OutputMap

    
`hist3d(arr, bins)`
:   

    
`matlab_style_gauss2D(shape=(3, 3), sigma=0.5)`
:   2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])