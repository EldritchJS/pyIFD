import numpy as np
from scipy.ndimage import correlate
import cv2
#from cagiutils import mat2gray


def characterizeblocks(MeanContent2,MeanStrongEdge, V_im, blk_idx,blk_idy, MeanInSpace,diff_Mean_Best_scaled,diff_Mean_Best_scaledInv,sgrid,PossiblePoints,kx,ky):

    uniform=np.zeros((int(np.floor(blk_idx/sgrid)),int(np.floor(blk_idy/sgrid))))

    for a in range(kx):
        for b in range(ky):
            for pp in range(16):
                if (MeanInSpace[a,b,pp]<(np.mean(MeanInSpace[:,:,pp]) *0.2)):
                    uniform[a,b]+=1
    
    st=np.std(np.reshape(uniform,(uniform.size,1),'F'))
    H =  np.ones((5,5))*0.04
    
    I=correlate(uniform,H,mode='constant')
    meanv=np.mean(I)             


    bg=0;
    for f in range(16):
        if ((PossiblePoints[f,0]==4) and (PossiblePoints[f,1]==4)):
            bg=f+1
      
    if bg==16:
        bestgrid=mat2gray(correlate(MeanInSpace[:,:,15],H,mode='constant'))
    elif bg==0:            # TODO find replacement for find(), this may not be it
        bg1= np.where(PossiblePoints[:,5]==max(PossiblePoints[:,5]))  
        bg=np.max(bg1)
        bestgrid=mat2gray(correlate(MeanInSpace[:,:,bg],H,mode='constant'))
    else:          
        bestgrid=mat2gray(correlate(MeanInSpace[:,:,bg],H,mode='constant'))
            
#//////////block based homogenous
    if ((np.mean(PossiblePoints[:,4])>0.4)  or (bg!=16)):
        homB=0
    else:
        homB=1
  
    if ((st/meanv)>1.5):
        I[I<(meanv+(st))]=0
        I[I>=(meanv+(st))]=homB
    else:
        I[I<(meanv+(st)/2)]=0
        I[I>=(meanv+(st)/2)]=homB


#/////////no content////////////////////////


    contentsc=(MeanContent2)

    x24=np.floor(blk_idx/3)
    y24=np.floor(blk_idy/3)

    hom=np.zeros((kx,ky))
    held=np.mean(contentsc)
    for i in range(kx):
        for j in range(ky):
            if (contentsc[i,j]<=4): #very soft responses
                hom[i,j]=1

    c=sgrid
    MeanStrongEdge2=np.zeros((kx,ky))
    for i in range(kx):
        for j in range(ky):
            a=i*sgrid
            b=j*sgrid
            MeanStrongEdge2[i,j]=np.mean(MeanStrongEdge[a:a+c, b:b+c])

    cc=8*sgrid
    V_im2=np.zeros((kx,ky))
    for i in range(kx):
        for j in range(ky):
            a=i*8*sgrid
            b=j*8*sgrid
            V_im2[i,j]=np.mean(V_im[a:a+cc, b:b+cc])

    V_imOver=V_im2;
    V_imUndr=V_im2;
    V_imOver[V_imOver>=245]=300
    V_imOver[V_imOver!=300]=0
    V_imUndr[V_imUndr<15]=300
    V_imUndr[V_imUndr!=300]=0

    V_imOver=mat2gray(V_imOver)
    V_imUndr=mat2gray(V_imUndr)
    MeanStrongEdge2[MeanStrongEdge2<0.5]=0
    MeanStrongEdge2[MeanStrongEdge2>=0.5]=1

#/////////////end overexposed/iunder and contours////////////////////

          
    touse=kx*ky
    notuse=np.zeros((kx,ky))
    for i in range(kx):
        for j in range(ky):
      
            if hom[i,j]==1:
                notuse[i,j]=3
            if MeanStrongEdge2[i,j]==1:
                notuse[i,j]=2         
            if ((V_imUndr[i,j]==1) or (V_imOver[i,j]==1)):
                notuse[i,j]=1

    for i in range(kx):
        for j in range(ky):   
            if notuse[i,j]==1:
                I[i,j]=1
    
    notused=np.sum(notuse[:]!=0)
    touse=kx*ky-notused
#//////////////excl NaN

    if touse==0:
        for i in range(kx):
            for j in range(ky):
                if  hom[i,j]==1 and I[i,j]!=1:
                    notuse[i,j]=0

    diff_Mean_Best_scaled_temp=diff_Mean_Best_scaled.copy()
    diff_Mean_Best_scaled_tempInv=diff_Mean_Best_scaledInv.copy()
    
    for a in range(int(np.floor(blk_idx/sgrid))):
        for b in range(int(np.floor(blk_idy/sgrid))):
            if I[a,b]==1:
                diff_Mean_Best_scaled_temp[a,b]=0
                diff_Mean_Best_scaled_tempInv[a,b]=1
            if diff_Mean_Best_scaled_temp[a,b]<np.mean(diff_Mean_Best_scaled) and homB==1:
                diff_Mean_Best_scaled_temp[a,b]=0
            if diff_Mean_Best_scaled_tempInv[a,b]<np.mean(diff_Mean_Best_scaledInv)and homB==1:
                diff_Mean_Best_scaled_tempInv[a,b]=1

    a+=1
    b+=1
    imageF = np.zeros((a,b))
    imageFInv = np.zeros((a,b))
    for x in range(a):
        for y in range(b):
            if x==0 or x==a-1 or y==0 or y==b-1:
                imageF[x,y]=diff_Mean_Best_scaled_temp[x,y]*(bestgrid[x,y])
            else:
                imageF[x,y]=diff_Mean_Best_scaled_temp[x,y]*(1-bestgrid[x,y])
            imageFInv[x,y]=diff_Mean_Best_scaled_tempInv[x,y]*(1-bestgrid[x,y])
         
    E_nofilt=imageF
    E=correlate(imageF, H,mode='constant')
          
    E_nofiltInv=imageFInv
    EInv=correlate(imageFInv, H,mode='constant')
    #Good through here w E_nofilt, E, E_nofiltInv, EInv, bestgrid, diff_Mean_Best_scaled_temp, and diff_Mean_Best_scaled_tempInv
# /////////////content based filtering//////////
    #notuse is different here!!
    uninteresting=np.zeros((touse,1))
    uninterestingInv=np.zeros((touse,1))
    a=-1
    for i in range(kx):
        for j in range(ky):
            if(notuse[i,j]==0):
                a+=1
                uninteresting[a]=E[i,j]
                uninterestingInv[a]=EInv[i,j] 
    MeanBlocksre=E_nofilt
    MeanBlocksreInv=E_nofiltInv
    meanuninteresting=np.mean(uninteresting)
    meanuninterestingInv=np.mean(uninterestingInv)
    for i in range(kx):
        for j in range(ky):
            if ((I[i,j]==1) and (notuse[i,j]==2)):
                I[i,j]=0
            if ((notuse[i,j]==1) or (MeanBlocksre[i,j]<meanuninteresting)):
                MeanBlocksre[i,j]=meanuninteresting
            if (((I[i,j]==1) and (MeanBlocksre[i,j]<meanuninteresting )) or  ((notuse[i,j]==3) and (I[i,j]==1))):
                MeanBlocksre[i,j]=meanuninteresting
            if ((notuse[i,j]==1) or (MeanBlocksreInv[i,j]>meanuninterestingInv)):
                MeanBlocksreInv[i,j]=meanuninterestingInv
            if (((I[i,j]==1)  and (MeanBlocksreInv[i,j]>meanuninterestingInv)) or ((notuse[i,j]==3) and (I[i,j]==1))):
                MeanBlocksreInv[i,j]=meanuninterestingInv
    E=correlate(MeanBlocksre, H,mode='reflect') 
    EInv=correlate(MeanBlocksreInv, H,mode='reflect') 
            
    return [E, EInv]
