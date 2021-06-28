from PIL import Image
from skimage.color import rgb2gray
import numpy as np

def im2double(im):
    info = np.iinfo(im.dtype) 
    return im.astype(np.double) / info.max 

#99% similarity, difference is with the first line resize.
def ImageTiling(OImg):
    Img = np.array(Image.fromarray(OImg.astype(np.uint8)).resize(size=(600,600),resample=Image.NEAREST))
    #[d1,d2]=Img.shape[0:2]
    #if (d2>601):      
    R1=rgb2gray(Img)
    R=R1*255#(im2double(R1)*255) 
    #else:
        #R=(im2double(Img)*255)


    blocks=3600
    stepX=60
    stepY=60
    ImgR=R.astype('int')

    countx=-1
    tile=np.zeros((10,10,blocks))
    for a in range(stepX):
        for b in range(stepY):
            countx+=1
            i=-1
            for x in range((a*10),(a*10)+10):
                i+=1
                j=-1
                for y in range((b*10),(b*10)+10):
                    j+=1;
                    tile[i,j,countx]=ImgR[x,y]
                    
    return tile

