"""
This module provides the ADQ2 Algorithm
"""


import numpy as np
import math
from scipy.signal import medfilt2d
import matplotlib.pyplot as plt
import scipy.io as spio
from skimage.metrics import structural_similarity as comp
import jpegio as jio

def bdctmtx(n):
    """Generates bdct matrix of size nxn"""
    [c,r]=np.meshgrid(range(8),range(8))
    [c0,r0]=np.meshgrid(r,r)
    [c1,r1]=np.meshgrid(c,c)
    x=np.zeros(np.shape(c))
    for i in range(n):
        for j in range(n):
            x[i,j]=math.sqrt(2/n)*math.cos(math.pi*(2*c[i,j]+1)*r[i,j]/(2*n))
    x[0,:]=x[0,:]/math.sqrt(2)
    x=x.flatten('F')
    m=np.zeros(np.shape(r0))
    for i in range(n**2):
        for j in range(n**2):
            m[i,j]=x[r0[i,j]+c0[i,j]*n]*x[r1[i,j]+c1[i,j]*n]
    return m

def im2vec(im, bsize, padsize=0):
    """Converts image to vector"""
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

def vec2im(v,padsize=[0,0],bsize=None,rows=None,cols=None):
    """Converts vector to image"""
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

def ibdct(a,n=8):
    """Generates inverse bdct matrix of size nxn"""
    dctm=bdctmtx(n)
    
    [v,r,c]=im2vec(a,n)
    b=vec2im(dctm.T @v,0,n,r,c)
    return b

def dequantize(qcoef,qtable):
    """Dequantizes a coef table given a quant table"""
    
    blksz=np.shape(qtable)
    [v,r,c]=im2vec(qcoef,blksz)
    
    flat=np.array(qtable).flatten('F')
    vec=v*np.tile(flat,(np.shape(v)[1],1)).T
    
    coef=vec2im(vec,0,blksz,r,c)
    return coef

def jpeg_rec(image):
    """Simulate decompressed JPEG image from JPEG object"""
    
    Y=ibdct(dequantize(image.coef_arrays[0],image.quant_tables[0]))
    Y+=128
    
    if(image.image_components==3):
        if(len(image.quant_tables)==1):
            image.quant_tables[1]=image.quant_tables[0]
            image.quant_tables[2]=image.quant_tables[0]
    
        Cb=ibdct(dequantize(image.coef_arrays[1],image.quant_tables[1]))
        Cr=ibdct(dequantize(image.coef_arrays[2],image.quant_tables[1]))
        
        [r,c]=np.shape(Y)
        [rC,cC]=np.shape(Cb)
        
        if(math.ceil(r/rC)==2) and (math.ceil(c/cC)==2): #4:2:0
            kronMat=np.ones((2,2))
        elif(math.ceil(r/rC)==1) and (math.ceil(c/cC)==4): #4:1:1
            kronMat=np.ones((1,4))
        elif(math.ceil(r/rC)==1) and (math.ceil(c/cC)==2): #4:2:2
            kronMat=np.ones((1,4))
        elif(math.ceil(r/rC)==1) and (math.ceil(c/cC)==1): #4:4:4
            kronMat=np.ones((1,1))
        elif(math.ceil(r/rC)==2) and (math.ceil(c/cC)==1): #4:4:0
            kronMat=np.ones((2,1))
        else:
            raise Exception("Subsampling method not recognized: "+str(np.shape(Y))+" "+str(np.shape(Cr)))
        
        Cb=np.kron(Cb,kronMat)+128
        Cr=np.kron(Cr,kronMat)+128
        
        Cb=Cb[:r,:c]
        Cr=Cr[:r,:c]
        
        I=np.zeros((r,c,3))
        I[:,:,0]=(Y+1.402*(Cr-128))
        I[:,:,1]=(Y-0.34414*(Cb-128)-0.71414*(Cr-128))
        I[:,:,2]=(Y+1.772*(Cb-128))
        YCbCr=np.concatenate((Y,Cb,Cr),axis=1)
    else:
        I=np.tile(Y,[1,1,3])
        YCbCr=cv2.cvtColor(I, cv2.COLOR_BGR2YCR_CB)
        
    return [I,YCbCr]

def bdct(a,n=8):
    """Applies bdct to block a of size nxn"""
    dctm=bdctmtx(n)
    
    [v,r,c]=im2vec(a,n)
    b=vec2im(dctm @ v,0,n,r,c)
    return b
    
def floor2(x1):
    """Applies floor to vector x1, but if an element is close to an integer, it is lowered by 0.5"""
    tol=1e-12
    x2=np.floor(x1)
    idx=np.where(np.absolute(x1-x2)<tol)
    x2[idx]=x1[idx]-0.5
    return x2

def ceil2(x1):
    """Applies ceil to vector x1, but if an element is close to an integer, it is raised by 0.5"""
    tol=1e-12
    x2=np.ceil(x1)
    idx=np.where(np.absolute(x1-x2)<tol)
    x2[idx]=x1[idx]+0.5
    return x2

def getJmap(filename, ncomp=1,c1=1,c2=15):
    """Main driver for ADQ2 algorithm. Input image of type jpg in filename"""
    try:
        image=jio.read(filename)
    except:
        return
    ncomp-=1#indexing
    coeffArray=image.coef_arrays[ncomp]
    qtable=image.quant_tables[image.comp_info[ncomp].quant_tbl_no]
    
    #estimate rounding and truncation error
    I=jpeg_rec(image)[0]
    Iint=I.copy()
    Iint[Iint<0]=0
    Iint[Iint>255]=255
    E=I-np.double(np.uint8(Iint+0.5))
    Edct=bdct(0.299*E[:,:,0]+0.587*E[:,:,1]+0.114*E[:,:,2])
    Edct2=np.reshape(Edct,(1,np.size(Edct)),order='F').copy()
    varE=np.var(Edct2)
    
    # simulate coefficients without DQ effect
    Y=ibdct(dequantize(coeffArray,qtable))
    coeffArrayS=bdct(Y[1:,1:])

    sizeCA=np.shape(coeffArray)
    sizeCAS=np.shape(coeffArrayS)
    coeff=[1, 9, 2, 3, 10, 17, 25, 18, 11, 4, 5, 12, 19, 26, 33, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 57, 50, 43, 36, 29, 22, 15, 8, 16, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 24, 32, 39, 46, 53, 60, 61, 54, 47, 40, 48, 55, 62, 63, 56, 64]
    coeffFreq=np.zeros((int(np.size(coeffArray)/64),1))
    coeffSmooth=np.zeros((int(np.size(coeffArrayS)/64),1))
    errFreq=np.zeros((int(np.size(Edct)/64),1))
    
    bppm=0.5*np.ones((int(np.size(coeffArray)/64),1))
    bppmTampered=0.5*np.ones((int(np.size(coeffArray)/64),1))
    
    q1table=100*np.ones(np.shape(qtable))
    alphatable=np.ones(np.shape(qtable))
    Q1up=np.concatenate((20*np.ones(10), 30*np.ones(5),40*np.ones(6),64*np.ones(7),80*np.ones(8),88*np.ones(28)))
    
    rangeC=np.arange(c1-1,c2)
    for index in rangeC:
        coe=coeff[index]
        #load DCT coefficients at position index
        k=0
        start=coe%8
        if(start==0):
            start=8
        rangeL=np.arange(start-1,sizeCA[1],8)
        rangeI=np.arange(math.ceil(coe/8)-1,sizeCA[0],8)
        for l in rangeL:
            for i in rangeI:
                coeffFreq[k]=coeffArray[i,l]
                errFreq[k]=Edct[i,l]
                k+=1
        k=0
        rangeL=np.arange(start-1,sizeCAS[1],8)
        rangeI=np.arange(math.ceil(coe/8)-1,sizeCAS[0],8)
        for l in rangeL:
            for i in rangeI:
                coeffSmooth[k]=coeffArrayS[i,l]
                k+=1
        
        #get histogram of DCT coefficients
        binHist=np.arange(-2**11,2**11-1)+0.5
        binHist=np.append(binHist,max(2**11,coeffFreq.max()))
        binHist=np.insert(binHist,0,min(-2**11,coeffFreq.min()))
        num4Bin=np.histogram(coeffFreq,binHist)[0]
        
        #get histogram of DCT coeffs w/o DQ effect (prior model for
        #uncompressed image
        Q2=qtable[math.floor((coe-1)/8),(coe-1)%8]
        binHist=np.arange(-2**11,2**11-1)+0.5
        binHist*=Q2
        binHist=np.append(binHist,max(Q2*(2**11),coeffSmooth.max()))
        binHist=np.insert(binHist,0,min(Q2*(-2**11),coeffSmooth.min()))
        hsmooth=np.histogram(coeffSmooth,binHist)[0]
        
        #get estimate of rounding/truncation error
        biasE=np.mean(errFreq)
        
        #kernel for histogram smoothing
        sig=math.sqrt(varE)/Q2
        f=math.ceil(6*sig)
        p=np.arange(-f,f+1)
        g=np.exp(-p**2/sig**2/2)
        g=g/sum(g)
        
        binHist=np.arange(-2**11,2**11)
        lidx=np.invert([binHist[i]!=0 for i in range(len(binHist))])
        hweight=0.5*np.ones((1,2**12))[0]
        E=float('inf')
        Etmp=np.ones((1,99))[0]*float('inf')
        alphaest=1
        Q1est=1
        biasest=0
        
        if(index==0):
            bias=biasE
        else:
            bias=0
        #estimate Q-factor of first compression
        rangeQ=np.arange(1,Q1up[index]+1)
        for Q1 in rangeQ:
            for b in [bias]:
                alpha=1
                if(Q2%Q1==0):
                    diff=np.square(hweight* (hsmooth-num4Bin))
                else:
                    #nhist * hsmooth = prior model for doubly compressed coefficient
                    nhist=Q1/Q2*(floor2((Q2/Q1)*(binHist+b/Q2+0.5))-ceil2((Q2/Q1)*(binHist+b/Q2-0.5))+1)
                    nhist=np.convolve(g,nhist)
                    nhist=nhist[f:-f]
                    a1=np.multiply(hweight,np.multiply(nhist,hsmooth)-hsmooth)
                    a2=np.multiply(hweight,hsmooth-num4Bin)
                    #Exclude zero bin from fitting
                    la1=np.ma.masked_array(a1,lidx).filled(0)
                    la2=np.ma.masked_array(a2,lidx).filled(0)
                    alpha=(-(la1 @ la2.T))/(la1 @ la1.T)
                    alpha=min(alpha,1)
                    diff=(hweight*(alpha*a1+a2))**2
                KLD=sum(np.ma.masked_array(diff,lidx).filled(0))
                if KLD<E and alpha >0.25:
                    E=KLD.copy()
                    Q1est=Q1.copy()
                    alphaest=alpha
                if KLD<Etmp[int(Q1)-1]:
                    Etmp[int(Q1)-1]=KLD
                    biasest=b
        Q1=Q1est.copy()
        nhist=Q1/Q2 * (floor2((Q2/Q1)*(binHist+biasest/Q2+0.5))-ceil2((Q2/Q1)*(binHist+biasest/Q2-0.5))+1)
        nhist=np.convolve(g,nhist)
        nhist=nhist[f:-f]
        nhist=alpha *nhist+1-alpha
        
        ppt=np.mean(nhist) / (nhist+np.mean(nhist))
        alpha=alphaest
        q1table[math.floor((coe-1)/8),(coe-1)%8]=Q1est
        alphatable[math.floor((coe-1)/8),(coe-1)%8]=alpha
        #compute probabilities if DQ effect is present
        if(Q2%Q1est>0):
            #index
            nhist=Q1est/Q2*(floor2((Q2/Q1est)*(binHist+biasest/Q2+0.5))-ceil2((Q2/Q1est)*(binHist+biasest/Q2-0.5))+1)
            #histogram smoothing (avoids false alarms)
            nhist=np.convolve(g,nhist)
            nhist=nhist[f:-f]
            nhist=alpha *nhist+1-alpha
            ppu=nhist/(nhist+np.mean(nhist))
            ppt=np.mean(nhist)/(nhist+np.mean(nhist))
            #set zeroed coefficients as non-informative
            ppu[2**11]=0.5
            ppt[2**11]=0.5
            idx=np.floor(coeffFreq+2**11).astype(int)
            bppm=bppm * ppu[idx]
            bppmTampered=bppmTampered * ppt[idx]            
    maskTampered=bppmTampered / (bppm+bppmTampered)
    maskTampered=np.reshape(maskTampered,(int(sizeCA[0]/8),int(sizeCA[1]/8)),order='F')
    #apply median filter to highlight connected regions
    maskTampered=medfilt2d(maskTampered,[5,5])
    return [maskTampered, q1table,alphatable]





