from numpy.matlib import repmat
import numpy as np
import jpegio as jio
import numpy as np
import math
import os

SupportVector = np.load(os.path.join(os.path.dirname(__file__),'SupportVector.npy'),allow_pickle=True)
AlphaHat = np.load(os.path.join(os.path.dirname(__file__),'AlphaHat.npy'),allow_pickle=True)
bias = np.array([ 0.10431149, -0.25288239, -0.2689174 ,  0.39425104, -1.11269764, -1.15730589, -1.18658372, -0.9444815 , -3.46445309, -2.9434976 ])

def BenfordDQ(filename):
    """Main driver of ADQ3. Required that filename is a jpg"""
    try:
        im=jio.read(filename)
    except:
        return
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
            Dist=svmdecision(Features,QualityInd-1) #SVMStruct[QualityInd-1][0][0])
            OutputMap[int(np.ceil((StartX-1)/Step)),int(np.ceil((StartY-1)/Step))]=Dist
    OutputMap=np.concatenate((repmat(OutputMap[0,:],int(np.ceil(BlockSize/2/Step)),1),OutputMap),axis=0)
    OutputMap=np.concatenate((repmat(OutputMap[:,0],int(np.ceil(BlockSize/2/Step)),1).T,OutputMap),axis=1)
    return OutputMap

def EstimateJPEGQuality(imIn):
    """Estimates the quality of JPEG imIn (0-100)"""
    if(len(imIn.quant_tables)==1):
        imIn.quant_tables[1]=imIn.quant_tables[0]
    YQuality=100-(np.sum(imIn.quant_tables[0])-imIn.quant_tables[0][0][0])/63
    CrCbQuality=100-(np.sum(imIn.quant_tables[1])-imIn.quant_tables[0][0][0])/63
    Diff=abs(YQuality-CrCbQuality)*0.98
    Quality=(YQuality+2*CrCbQuality)/3+Diff
    return Quality

def ExtractFeatures(im,c1,c2,ncomp,digitBinsToKeep):
    """This function extracts a descriptor feature based on the first-digit distribution of DCT coefficients of an image. It is needed by BenfordDQ. 
    
     c1 and c2 are the first and last DCT coefficients to be taken into account, DC term included (default: c1=2, c2=10). ncomp is the component from which to extract the feature (default: 1, which corresponds to the Y component digitBinsToKeep is an array containing the digits for which we keep their frequency. Default digitBinsToKeep=[2 5 7]"""
    coeffArray=im.coef_arrays[ncomp-1]
    qtable=im.quant_tables[im.comp_info[ncomp].quant_tbl_no-1]
    Y=dequantize(coeffArray,qtable)
    coeff=[1, 9, 2, 3, 10, 17, 25, 18, 11, 4, 5, 12, 19, 26, 33, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 57, 50, 43, 36, 29, 22, 15, 8, 16, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 24, 32, 39, 46, 53, 60, 61, 54, 47, 40, 48, 55, 62, 63, 56, 64]
    sizeCA=np.shape(coeffArray)
    digitHist=np.zeros((c2-c1+1,10))
    for index in range(c1,c2+1):
        coeffFreq=np.zeros((int(np.size(coeffArray)/64),))
        coe=coeff[index-1]
        k=1
        start=coe%8
        if start==0:
            start=8
        for l in range(start,sizeCA[1]+1,8):
            for i in range(int(np.ceil(coe/8)),sizeCA[0],8):
                coeffFreq[k-1]=Y[i-1,l-1]
                k+=1
        NumOfDigits=(np.floor(np.log10(abs(coeffFreq)+0.5))+1)
        tmp=[10**(i-1) for i in np.array(NumOfDigits)]
        FirstDigit=np.floor(np.divide(abs(coeffFreq),tmp)).astype("uint8")
        
        binHist=list(np.arange(0.5,9.5,1))
        binHist.insert(0,-float('Inf'))
        binHist.append(float('Inf'))
        digitHist[index-c1,:]=np.histogram(FirstDigit,binHist)[0]
    HistToKeep=digitHist[:,digitBinsToKeep]
    return np.ndarray.flatten(HistToKeep)

def vec2im(v,padsize=[0,0],bsize=None,rows=None,cols=None):
    """Converts vector to an image"""
    [m,n]=np.shape(v)
    
    padsize=padsize+np.zeros((1,2),dtype=int)[0]
    if(padsize.any()<0):
        raise Exception("Pad size must not be negative")
    if(bsize==None):
        bsize=math.floor(math.sqrt(m))
    bsize=bsize+np.zeros((1,2),dtype=int)[0]
    
    if(np.prod(bsize)!=m):
        raise Exception("Block size does not match size of input vectors.")
    
    if(rows==None):
        rows=math.floor(math.sqrt(n))
    if(cols==None):
        cols=math.ceil(n/rows)
    
    #make image
    y=bsize[0]+padsize[0]
    x=bsize[1]+padsize[1]
    t=np.zeros((y,x,rows*cols))
    t[:bsize[0],:bsize[1],:n]=np.reshape(v,(bsize[0],bsize[1],n),order='F')
    t=np.reshape(t,(y,x,rows,cols),order='F')
    t=np.reshape(np.transpose(t,[0,2,1,3]),(y*rows,x*cols),order='F')
    im=t[:y*rows-padsize[0],:x*cols-padsize[1]]
    return im

def im2vec(im, bsize, padsize=0):
    """Converts image to a vector"""
    bsize=bsize+np.zeros((1,2),dtype=int)[0]
    padsize=padsize+np.zeros((1,2),dtype=int)[0]
    if(padsize.any()<0):
        raise Exception("Pad size must not be negative")
    imsize=np.shape(im)
    y=bsize[0]+padsize[0]
    x=bsize[1]+padsize[1]
    rows=math.floor((imsize[0]+padsize[0])/y)
    cols=math.floor((imsize[1]+padsize[1])/x)
    t=np.zeros((y*rows,x*cols))
    imy=y*rows-padsize[0]
    imx=x*cols-padsize[1]
    t[:imy,:imx]=im[:imy,:imx]
    t=np.reshape(t,(y,rows,x,cols),order='F')
    t=np.reshape(np.transpose(t,[0,2,1,3]),(y,x,rows*cols),order='F')
    v=t[:bsize[0],:bsize[1],:rows*cols]
    v=np.reshape(v,(y*x,rows*cols),order='F')
    return [v,rows,cols]

def dequantize(qcoef,qtable):
    """Dequantizes a coef table given a quant table"""
    blksz=np.shape(qtable)
    [v,r,c]=im2vec(qcoef,blksz)
    
    flat=np.array(qtable).flatten('F')
    vec=v*np.tile(flat,(np.shape(v)[1],1)).T
    
    coef=vec2im(vec,0,blksz,r,c)
    return coef

def svmdecision(Xnew,index):
    """Uses given index of svm to classify Xnew"""
    f=np.dot(np.tanh(SupportVector[index] @ np.transpose(Xnew)-1),AlphaHat[index])+bias[index]
    return f
    
