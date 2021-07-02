import scipy.io as spio
from numpy.matlib import repmat
import numpy as np
#from svmdecision import svmdecision
#from ExtractFeatures import ExtractFeatures
from pyIFD.ADQ3.EstimateJPEGQuality import EstimateJPEGQuality
def ADQ3(im):
    out=spio.loadmat("../SVMs.mat")
    SVMStruct=out['SVMStruct'][0]
    Quality=EstimateJPEGQuality(im)
    QualityInd=int(np.round((Quality-50)/5+1))
    if QualityInd>10:
        QualityInd=10
    elif QualityInd<1:
        QualityInd=1
    c1=2
    c2=10
    ncomp=1
    digitBinsToKeep=[2,5,7]
    block=im
    qtable=im.quant_tables[im.comp_info[ncomp].quant_tbl_no-1]
    YCoef=im.coef_arrays[ncomp-1]
    Step=8
    BlockSize=64
    maxX=np.shape(YCoef)[0]+1-BlockSize
    maxY=np.shape(YCoef)[1]+1-BlockSize
    OutputMap=np.zeros((int(np.ceil(maxX-1)/Step+1),int(np.ceil(maxY-1)/Step+1)))
    if np.shape(im.coef_arrays[0])[0]<BlockSize:
        return 0
    
    for X in range(1,np.shape(YCoef)[0]+1,Step):
        StartX=min(X,np.shape(YCoef)[0]-BlockSize+1)
        for Y in range(1,np.shape(YCoef)[1]+1,Step):
            StartY=min(Y,np.shape(YCoef)[1]-BlockSize+1)
            block.coef_arrays[ncomp-1]=YCoef[StartX-1:StartX+BlockSize-1,StartY-1:StartY+BlockSize-1]
            Features=ExtractFeatures(block,c1,c2,ncomp,digitBinsToKeep)
            Features/=64
            Dist=svmdecision(Features,SVMStruct[QualityInd-1][0][0])
            OutputMap[int(np.ceil((StartX-1)/Step)),int(np.ceil((StartY-1)/Step))]=Dist
    OutputMap=np.concatenate((repmat(OutputMap[0,:],int(np.ceil(BlockSize/2/Step)),1),OutputMap),axis=0)
    OutputMap=np.concatenate((repmat(OutputMap[:,0],int(np.ceil(BlockSize/2/Step)),1).T,OutputMap),axis=1)
    return OutputMap
