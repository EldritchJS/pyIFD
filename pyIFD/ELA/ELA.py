import numpy as np
import cv2
import os

def ELA(Filename, Quality=90, Multiplier=15, Flatten=True):
    ImIn=np.double(cv2.imread(Filename))
    cv2.imwrite('tmpResave.jpg', ImIn, [cv2.IMWRITE_JPEG_QUALITY, Quality])
    ImJPG = np.double(cv2.imread('tmpResave.jpg'))

    OutputMap=(np.abs(ImIn-ImJPG))*Multiplier
    OutputMap[:,:,[0,2]] = OutputMap[:,:,[2,0]]

    if Flatten==True:
        OutputMap=np.mean(OutputMap,2)

    os.remove('tmpResave.jpg')
    return OutputMap

