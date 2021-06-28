from scipy import signal
from skimage.transform import resize
import numpy as np
import math
import cv2
import jpeg2dct as j2dct
from inblockpatterns import inblockpatterns
from predict0 import predict0
from predict1 import predict1
from MainTrain import MainTrain
from characterizeblocks import characterizeblocks
from RescaleToImageResult import RescaleToImageResult
from scores_pick_variables import scores_pick_variables

def CAGI(filename='../demo.jpg'):
    # Read image in as double RGB
    BGR=cv2.imread(filename)
    RGB=np.double(BGR[...,::-1])
    
    (height,width,color) = RGB.shape
    V_im=cv2.cvtColor(np.uint8(RGB), cv2.COLOR_RGB2HSV)[:,:,2]
    
    # Store the pixels for Y of YCbCr
    pixels=16/255+ (0.25678823529411759496454692452971 * RGB[:,:,0] + 0.50412941176470582593793778869440 * RGB[:,:,1] + 0.09790588235294117591678286771639 * RGB[:,:,2])
    
    if ((height*width)<(480*640)):
        sgrid=2
    else:
        sgrid=3
    bins=40
    imageGS=pixels
    (x,y)=imageGS.shape
    blk_idx = int(np.floor((x/8)-1))
    blk_idy = int(np.floor((y/8)-1))
    kx = int(np.floor(blk_idx/sgrid))
    ky = int(np.floor(blk_idy/sgrid))
    BlockScoreAll = np.zeros((blk_idx,blk_idy, 8, 8))
    Kscores = np.zeros((8,8,2))
    #K=0
    #Correct=0
    #All good through here
    for p in range(8):
        for q in range(8):
            #Slight error w/i inblockpatterns 
            (K, Correct, BlockScoreAll[:, :, p, q]) = inblockpatterns(imageGS, bins, p+1, q+1, blk_idx, blk_idy)
            if (K>1.999999):
                Kscores[p, q, 0]=0
            else:
                Kscores[p,q,0]=K
            Kscores[p, q, 1]=Correct

    [Kpredict, Kpre] = predict0(Kscores)
    PossiblePoints = predict1(Kscores, Kpredict, Kpre)
    PossiblePoints = PossiblePoints[np.argsort(PossiblePoints[:,6])] 
    #Slightly different results also in MainTrain->ImageTiling
    [MeanContent, MeanStrongEdge] = MainTrain(RGB, blk_idx, blk_idy)
    MeanContent2 = np.zeros((kx,ky))
    for i in range(kx):
        for j in range(ky):
            a = i*sgrid
            b=j*sgrid
            ccc=sgrid
            MeanContent2[i,j]=np.mean(MeanContent[a:a+ccc, b:b+ccc])
    [MeanInSpace,PossiblePoints,diff_Mean_Best_scaled,diff_Mean_Best_scaledInv]=scores_pick_variables(BlockScoreAll,sgrid,blk_idx,blk_idy,PossiblePoints,kx,ky)
    
    [E,EInv]=characterizeblocks(MeanContent2,MeanStrongEdge, V_im, blk_idx,blk_idy, MeanInSpace,diff_Mean_Best_scaled,diff_Mean_Best_scaledInv,sgrid,PossiblePoints,kx,ky)            
    Result_CAGI=RescaleToImageResult(E,sgrid,kx,ky,pixels)
    Result_Inv_CAGI=RescaleToImageResult(EInv,sgrid,kx,ky,pixels)
    
    return [Result_CAGI, Result_Inv_CAGI]



