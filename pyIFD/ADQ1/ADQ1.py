import jpegio as jio
import matplotlib.image as mpimg
from detectDQ_JPEG import detectDQ_JPEG
from detectDQ_NonJPEG import detectDQ_NonJPEG
import numpy as np
def ADQ1( impath ):
    if impath[-4:]==".jpg":
        [OutputMap, Feature_Vector, coeffArray] = detectDQ_JPEG( jio.read(impath) )
    else:
        im=mpimg.imread(impath)
        im=np.round(im*255)
        [OutputMap, Feature_Vector, coeffArray] = detectDQ_NonJPEG( im )
    return [OutputMap, Feature_Vector, coeffArray] 



