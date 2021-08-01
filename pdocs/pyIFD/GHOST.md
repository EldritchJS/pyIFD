Module pyIFD.GHOST
==================
This module provides the GHOST algorithm

Functions
---------

    
`GHOST(impath, checkDisplacements=0)`
:   Main driver for GHOST algorithm.
    
    Args:
        impath: Path to image to be transformed.
        checkDisplacements (0 or 1, optional, default=0): whether to run comparisons for all 8x8 displacements in order to find the NA-match.
    
    Returns:
        OutputX:
        OutputY:
        dispImages: Equivalent of OutputMap.
        imin:
        Qualities:
        Mins:
    
    TODO:
    Find purpose of other outputs, and if they are needed.