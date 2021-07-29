"""
This module provides the ELA algorithm
"""

import numpy as np
import cv2
import os

def ELA(impath, Quality=90, Multiplier=15, Flatten=True):
    """
    Main driver for ELA algorithm.
    
    Args:
        impath: Path to image to be transformed.
        Quality (optional, default=90): the quality in which to recompress the image. (0-100 integer).
        Multiplier (optional, default=15): value with which to multiply the residual to make it more visible. (Float).
        Flatten (optional, default=True): Boolean. Describes whether to flatten OutputMap.

    Returns:
        OutputMap: Output of ELA algorithm. 
    """
    ImIn=np.double(cv2.imread(impath))
    cv2.imwrite('tmpResave.jpg', ImIn, [cv2.IMWRITE_JPEG_QUALITY, Quality])
    ImJPG = np.double(cv2.imread('tmpResave.jpg'))

    OutputMap=(np.abs(ImIn-ImJPG))*Multiplier
    OutputMap[:,:,[0,2]] = OutputMap[:,:,[2,0]]

    if Flatten==True:
        OutputMap=np.mean(OutputMap,2)

    os.remove('tmpResave.jpg')
    return OutputMap

