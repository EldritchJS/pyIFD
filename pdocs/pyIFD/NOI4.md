Module pyIFD.NOI4
=================
This module provides the NOI4 algorithm

Functions
---------

    
`MedFiltForensics(impath, NSize=3, Multiplier=10, Flatten=True)`
:   Main driver for NOI4.
    
    Args:
        impath: input image
        NSize (optional, default=3): size of blocks to apply median filter to
        Multiplier: Number to scale output by
    Flatten: Whether to flatten output or not (False/True)
    
    Output args:
    OutputMap: Output image