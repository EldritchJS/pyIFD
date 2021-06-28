import numpy as np
from skimage.color import rgb2ycbcr
from PIL import Image


def GetNoiseMaps_ram( im, filter_type, filter_size, block_rad ):
    # Markos Zampoglou: This is the original version of the code, where all
    # processing takes place in memory
    
    YCbCr=np.double(rgb2ycbcr(im))
    im=np.round(YCbCr[:,:,0])
    
    flt = np.ones((filter_size,1))
    flt = (flt*np.transpose(flt))/(filter_size**2)
    noiIm = conv2(im,flt,'same')
    
    #estV_tmp = localNoiVarEstimate_ram(noiIm, filter_type, filter_size, block_rad)
    #estV = imresize(single(estV_tmp),round(size(estV_tmp)/4),'method','box')
    estV_tmp = localNoiVarEstimate_hdd(noiIm, filter_type, filter_size, block_rad)
    
    estV = imresize(single(estV_tmp),np.round(np.size(estV_tmp)/4),'method','box')  # TODO: is single necessary?

    estV[estV<=0.001]=np.mean(estV)

    return estV

