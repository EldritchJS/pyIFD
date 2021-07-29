"""
This module provides the NOI4 algorithm
"""

import numpy as np
from scipy.signal import medfilt
from PIL import Image


def MedFiltForensics(impath, NSize=3, Multiplier=10, Flatten=True):
    """
    Main driver for NOI4.

    Args:
        impath: input image
        NSize (optional, default=3): size of blocks to apply median filter to
        Multiplier: Number to scale output by
    Flatten: Whether to flatten output or not (False/True)

    Output args:
    OutputMap: Output image
    """
    Im = Image.open(impath)
    ImIn = np.array(Im, dtype=np.double)
    [x, y, channels] = ImIn.shape
    ImMed = np.zeros((x, y, channels))

    for Channel in range(channels):
        ImMed[:, :, Channel] = medfilt(ImIn[:, :, Channel], [NSize, NSize])

    OutputMap = (np.abs(ImIn-ImMed))*Multiplier

    if Flatten is True:
        OutputMap = np.mean(OutputMap, 2)

    return OutputMap
