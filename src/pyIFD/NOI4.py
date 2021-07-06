import numpy as np
from scipy.signal import medfilt
from PIL import Image

def MedFiltForensics(Filename, NSize=3, Multiplier=10, Flatten=True):

    Im=Image.open(Filename)
    ImIn=np.array(Im, dtype=np.double)
    [x,y,channels] = ImIn.shape
    ImMed=np.zeros((x,y,channels))
    
    for Channel in range(channels):
        ImMed[:,:,Channel]=medfilt(ImIn[:,:,Channel],[NSize,NSize])

    OutputMap = (np.abs(ImIn-ImMed))*Multiplier

    if Flatten==True:
        OutputMap = np.mean(OutputMap,2)

    return OutputMap
