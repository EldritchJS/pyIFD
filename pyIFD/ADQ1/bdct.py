import math
import numpy as np

def vec2im(v,padsize=[0,0],bsize=None,rows=None,cols=None):
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

def bdctmtx(n):
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

def bdct(a,n=8):
    dctm=bdctmtx(n)
    
    [v,r,c]=im2vec(a,n)
    b=vec2im(dctm @ v,0,n,r,c)
    return b