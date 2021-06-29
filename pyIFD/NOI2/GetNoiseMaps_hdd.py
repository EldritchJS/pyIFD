import numpy as np
from skimage.color import rgb2ycbcr
from PIL import Image
import PIL
from scipy.signal import convolve2d
#from localNoiVarEstimate_hdd import localNoiVarEstimate_hdd

def conv2(x, y, mode='same'):
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)

def GetNoiseMaps_hdd( image, filter_type, filter_size, block_rad ):
    # Markos Zampoglou: This a variant version of the code, which calls
    # localNoiVarEstimate_hdd, a version in which intermediate data are
    # stored on disk
    
    YCbCr=np.double(rgb2ycbcr(image))
    im=np.round(YCbCr[:,:,0])
    
    flt = np.ones((filter_size,1))
    flt = (flt*np.transpose(flt))/(filter_size**2)
    noiIm = conv2(im,flt,'same') 

    estV_tmp = localNoiVarEstimate_hdd(noiIm, filter_type, filter_size, block_rad)
    estV = np.array(Image.fromarray(estV_tmp).resize(np.flip(np.round(np.asarray(np.shape(estV_tmp))/4)).astype(int),resample=PIL.Image.BOX))     
    estV[estV<=0.001]=np.mean(estV)
    return estV

