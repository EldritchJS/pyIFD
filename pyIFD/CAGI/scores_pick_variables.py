import numpy as np
from cagiutils import mat2gray

def scores_pick_variables(BlockScoreALL,sgrid,blk_idx,blk_idy,PossiblePoints,kx,ky):

 
    BlockScore=np.zeros((blk_idx,blk_idy,16)) 
    for i in range(16):
        p=PossiblePoints[i,0]
        q=PossiblePoints[i,1]
        BlockScore[:,:,i]=BlockScoreALL[:,:,int(p-1),int(q-1)]/255

    MeanInSpace=np.zeros((kx,ky,16))
    for r in range(16):
        for i in range(kx):
            for j in range(ky):
                b=(i+1)*sgrid
                a=b-sgrid
                d=(j+1)*sgrid
                c=d-sgrid
                MeanInSpace[i,j,r]=np.mean(BlockScore[a:b,c:d,r])
               
    MeanOfAllGrids=np.mean(MeanInSpace,axis=2)
      
    BestGrid=MeanInSpace[:,:,15];
    diff_Mean_Best=MeanOfAllGrids - BestGrid
    diff_Mean_Best_scaled=mat2gray(diff_Mean_Best)
   
    bg=0;
    for f in range(16):
        if (PossiblePoints[f,0]==4 and PossiblePoints[f,1]==4):
            bg=f
            
    for f in range(16):
        if (bg==0):
            bg1= np.where(PossiblePoints[:,5]==max(PossiblePoints[:,5]))
            bg=np.max(bg1);
                   

    BestGridInv = np.zeros((kx,ky)) 
    BestGridInv=MeanInSpace[:,:,bg]
    diff_Mean_BestInv=MeanOfAllGrids - BestGridInv
    diff_Mean_Best_scaledInv=mat2gray(diff_Mean_BestInv)

    return [MeanInSpace,PossiblePoints,diff_Mean_Best_scaled,diff_Mean_Best_scaledInv]


