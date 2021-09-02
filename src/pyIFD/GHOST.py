"""
This module provides the GHOST algorithm

JPEG-block-artifact-based detector, solution 3 (leveraging JPEG ghosts).

Algorithm attribution:
Farid, Hany. "Exposing digital forgeries from JPEG ghosts." Information Forensics and Security, IEEE Transactions on 4, no. 1 (2009): 154-160.

Based on code from:
Zampoglou, M., Papadopoulos, S., & Kompatsiaris, Y. (2017). Large-scale evaluation of splicing localization algorithms for web images. Multimedia Tools and Applications, 76(4), 4801–4834.
"""

from scipy import signal
from scipy.signal import fftconvolve
from skimage.transform import resize
import numpy as np
import cv2


def GHOST(impath, checkDisplacements=0):
    """
    Main driver for GHOST algorithm.

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
    """
    imorig = np.double(cv2.imread(impath))
    minQ = 51
    maxQ = 100
    stepQ = 1
    dispImages = {}
    Output = np.zeros(int((maxQ-minQ)/stepQ+1))
    Mins = np.zeros(int((maxQ-minQ)/stepQ+1))

    if(checkDisplacements == 1):
        maxDisp = 7
    else:
        maxDisp = 0

    smoothing_b = 17
    h = (np.ones((smoothing_b, smoothing_b))/(smoothing_b**2))[:, :, None]
    Offset = int((smoothing_b-1)/2)

    for ii in range(minQ, maxQ+1, stepQ):
        encimg = cv2.imencode('.jpg', imorig, [int(cv2.IMWRITE_JPEG_QUALITY), ii])[1].tobytes()
        nparr = np.frombuffer(encimg, np.byte)
        tmpResave = np.double(cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR))
        Deltas = {}
        overallDelta = {}
        ComparisonFull = (tmpResave-imorig)**2
        ComparisonFull = np.double(fftconvolve(ComparisonFull, h, 'same'))
        for dispx in range(maxDisp+1):
            for dispy in range(maxDisp+1):
                DisplacementIndex = dispx*8+dispy
                Comparison=ComparisonFull[Offset+dispx:-Offset, Offset+dispy:-Offset]
                Deltas[DisplacementIndex] = np.mean(Comparison, axis=2)
                overallDelta[DisplacementIndex] = np.mean(Deltas[DisplacementIndex])
        minInd = min(overallDelta, key=overallDelta.get)
        minOverallDelta = overallDelta[minInd]
        Mins[int((ii-minQ)/stepQ)] = minInd+1  # Add 1 to acct for matlab starting idx counting at 1
        Output[int((ii-minQ)/stepQ)] = minOverallDelta
        delta = Deltas[minInd]
        delta = (delta-delta.min())/(delta.max()-delta.min())
        newSize = (round((delta.shape[i])/4) for i in range(2))
        dispImages[int((ii-minQ)/stepQ)] = resize(np.float32(delta), newSize)
    OutputX = range(minQ, maxQ+1, stepQ)
    OutputY = Output
    imin = signal.argrelextrema(OutputY, np.less)[0]+1  # Add one to acct for matlab starting idx counting at 1.
    if(OutputY[-1] < OutputY[-2]):  # Check last point
        imin = np.append(imin, len(OutputY))
    if(OutputY[0] < OutputY[1]):  # Check first point
        imin = np.insert(imin, 0, 1)
    Qualities = imin*stepQ+minQ-1
    return [OutputX, OutputY, dispImages, imin, Qualities, Mins]    