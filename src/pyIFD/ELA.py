"""
This module provides the ELA algorithm

Error-level-analysis-based detector.

Algorithm attribution:
Krawets, Neil. "A Picture's Worth: Digital Image Analysis and Forensics". Online
article on http://www.google.gr/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwiDg5_c07PLAhVpnXIKHUp8B5QQFgggMAA&url=http%3A%2F%2Fwww.hackerfactor.com%2Fpapers%2Fbh-usa-07-krawetz-wp.pdf&usg=AFQjCNFuUo7D6kGBAP9jAEmSgmY6RtWZ4w&sig2=Xw9SdzUHLYJ6dfPVzUmFLw

Based on code from:
Zampoglou, M., Papadopoulos, S., & Kompatsiaris, Y. (2017). Large-scale evaluation of splicing localization algorithms for web images. Multimedia Tools and Applications, 76(4), 4801–4834.
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
    ImIn = np.double(cv2.imread(impath))
    cv2.imwrite('tmpResave.jpg', ImIn, [cv2.IMWRITE_JPEG_QUALITY, Quality])
    ImJPG = np.double(cv2.imread('tmpResave.jpg'))

    OutputMap = (np.abs(ImIn-ImJPG))*Multiplier
    OutputMap[:, :, [0, 2]] = OutputMap[:, :, [2, 0]]

    if Flatten is True:
        OutputMap = np.mean(OutputMap, 2)

    os.remove('tmpResave.jpg')
    return OutputMap
