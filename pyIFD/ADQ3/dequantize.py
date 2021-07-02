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


import math
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

def dequantize(qcoef,qtable):
    
    blksz=np.shape(qtable)
    [v,r,c]=im2vec(qcoef,blksz)
    
    flat=np.array(qtable).flatten('F')
    vec=v*np.tile(flat,(np.shape(v)[1],1)).T
    
    coef=vec2im(vec,0,blksz,r,c)
    return coef