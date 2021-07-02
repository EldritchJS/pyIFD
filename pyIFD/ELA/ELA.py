import numpy as np
from cv2 import IMWRITE_JPEG_QUALITY,imread,imwrite
import os

def ELA(Filename, Quality=90, Multiplier=15, Flatten=True):
    ImIn=np.double(imread(Filename))
    imwrite('tmpResave.jpg', ImIn, [IMWRITE_JPEG_QUALITY, Quality])
    ImJPG = np.double(imread('tmpResave.jpg'))

    OutputMap=(np.abs(ImIn-ImJPG))*Multiplier
    OutputMap[:,:,[0,2]] = OutputMap[:,:,[2,0]]

    if Flatten==True:
        OutputMap=np.mean(OutputMap,2)

    os.remove('tmpResave.jpg')
    return OutputMap

