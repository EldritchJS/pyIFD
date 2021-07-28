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
        impath:
        Quality (optional, default=90):
        Multiplier (optional, default=15):
        Flatten (optional, default=True):

    Returns:
        OutputMap:
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

