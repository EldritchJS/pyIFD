from scipy import signal
from scipy.signal import fftconvolve
from skimage.transform import resize
import numpy as np
import math
from cv2 import imread,imwrite,IMWRITE_JPEG_QUALITY 
import os

# TODO: Check speedup using cv2 imencode decode

def GHOST(filename, checkDisplacements=0):
    imorig=np.double(imread(filename))
    minQ=51
    maxQ=100
    stepQ=1
    dispImages={}
    Output=np.zeros(int((maxQ-minQ)/stepQ+1))
    Mins=np.zeros(int((maxQ-minQ)/stepQ+1))
    
    
    if(checkDisplacements==1):
        maxDisp=7
    else:
        maxDisp=0
    
    smoothing_b=17
    Offset=int((smoothing_b-1)/2)
    
    for ii in range(minQ,maxQ+1,stepQ):
        imwrite('tmpResave.jpg', imorig, [IMWRITE_JPEG_QUALITY, ii])
        tmpResave=np.double(imread('tmpResave.jpg'))
        Deltas={}
        overallDelta={}
        for dispx in range(maxDisp+1):
            for dispy in range(maxDisp+1):
                DisplacementIndex=dispx*8+dispy
                tmpResave_disp=tmpResave[dispx:,dispy:,:]
                if(dispx==0 and dispy==0):#This if/elif statement removes the last dispx entries of the x-coord, and the last dispy entries of the y-coord
                    imorig_disp=imorig
                elif(dispx==0):
                    imorig_disp=imorig[:,0:-dispy,:]
                elif(dispy==0):
                    imorig_disp=imorig[0:-dispx,:,:]
                else:
                    imorig_disp=imorig[0:-dispx,0:-dispy,:]
                Comparison=(imorig_disp-tmpResave_disp)**2
                h=np.ones((smoothing_b,smoothing_b))/(smoothing_b**2)
                for jj in range(3):
                    Comparison[:,:,jj]=fftconvolve(Comparison[:,:,jj],h,mode='same')#2d convolution
                Comparison=Comparison[Offset:-Offset,Offset:-Offset,:]
                Deltas[DisplacementIndex]=np.mean(Comparison,axis=2)
                overallDelta[DisplacementIndex]=np.mean(Deltas[DisplacementIndex])
        
        minInd=min(overallDelta, key=overallDelta.get) 
        minOverallDelta=overallDelta[minInd]
        Mins[int((ii-minQ)/stepQ)]=minInd+1#Add 1 to acct for matlab starting idx counting at 1
        Output[int((ii-minQ)/stepQ)]=minOverallDelta
        delta=Deltas[minInd]
        delta=(delta-delta.min())/(delta.max()-delta.min())
        newSize=(round((delta.shape[i])/4) for i in range(2))
        dispImages[int((ii-minQ)/stepQ)]=resize(np.float32(delta),newSize)
        
    OutputX=range(minQ,maxQ+1,stepQ)
    OutputY=Output
    imin=signal.argrelextrema(OutputY, np.less)[0]+1#Add one to acct for matlab starting idx counting at 1.
    if(OutputY[-1]<OutputY[-2]):#Check last point
        imin=np.append(imin,len(OutputY))
    if(OutputY[0]<OutputY[1]):#Check first point
        imin=np.insert(imin,0,1)
    Qualities=imin*stepQ+minQ-1
    os.remove("tmpResave.jpg")
    return [OutputX,OutputY,dispImages, imin, Qualities,Mins]
