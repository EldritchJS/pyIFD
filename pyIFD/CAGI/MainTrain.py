from ImageTiling import ImageTiling
from SmapIng import SmapIng
from filtering import filtering
from filteringMethod import filteringMethod
from PaintimgEdges import PaintimgEdges
from PIL import Image
import numpy as np
import scipy.io as spio

def MainTrain(R10,blk_idx,blk_idy):
    [x,y,z]=R10.shape

    masks=spio.loadmat('masks.mat')
    PMasks=masks['PMasks']
    MMasks=masks['MMasks']
    MaskWhite=masks['MaskWhite']
    #////////Image Tiling 3 Scales////////////////////////////
    #slight difference in tileF (~99% similarity)
    tileF=ImageTiling(R10)

    #////////////Smaping/////////////////////////////////////
    smapF=SmapIng(tileF, PMasks, MaskWhite)
    #% % % %////////////Filtering///////////////////////////////////
    [ThresSmall,ThresBig, ThresImg] =filtering(smapF)
    smapF_filtrOld=filteringMethod(smapF, ThresSmall, ThresBig, ThresImg)
    #Through here so far
    #/////////////PaintEdges///////////////////////////////// This uses NN PIL using mean
    [e,edge,contours]=PaintimgEdges(smapF_filtrOld, MMasks, 1)
    Output = np.array(Image.fromarray(e.astype(np.double)).resize(size=(y,x),resample=Image.NEAREST))
    StrongEdge = np.array(Image.fromarray(contours.astype(np.uint8)).resize(size=(y,x),resample=Image.NEAREST))


    MeanContent = np.zeros((blk_idx,blk_idy))
    MeanStrongEdge = np.zeros((blk_idx,blk_idy))
    for i in range(blk_idx):
        for j in range(blk_idy):
            a=i*8
            b=j*8
            MeanContent[i,j]=np.mean(Output[a:a+8, b:b+8])
            MeanStrongEdge[i,j]=np.mean(StrongEdge[a:a+8, b:b+8])
    MeanStrongEdge[MeanStrongEdge>0.5]=1

    MeanStrongEdge[MeanStrongEdge<=0.5]=0 
        

    return [MeanContent,MeanStrongEdge]