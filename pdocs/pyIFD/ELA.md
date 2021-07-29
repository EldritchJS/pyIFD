Module pyIFD.ELA
================
This module provides the ELA algorithm

Functions
---------

    
`ELA(impath, Quality=90, Multiplier=15, Flatten=True)`
:   Main driver for ELA algorithm.
    
    Args:
        impath: Path to image to be transformed.
        Quality (optional, default=90): the quality in which to recompress the image. (0-100 integer).
        Multiplier (optional, default=15): value with which to multiply the residual to make it more visible. (Float).
        Flatten (optional, default=True): Boolean. Describes whether to flatten OutputMap.
    
    Returns:
        OutputMap: Output of ELA algorithm.