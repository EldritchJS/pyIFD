Module pyIFD.NOI1
=================
This module provides the NOI1 algorithm

Functions
---------

    
`GetNoiseMap(impath, BlockSize=8)`
:   Main driver for NOI1 algorithm.
    
    Args:
        impath: Path to the image to be processed.
        BlockSize: the block size for noise variance estimation. Too small reduces quality, too large reduces localization accuracy
    
    Returns:
        OutputMap: