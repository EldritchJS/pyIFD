import numpy as np
from numpy import sum as npsum
from numpy import sum as npmax
from numpy import sum as npmin
from skimage.color import rgb2ycbcr
from scipy.signal import medfilt
from scipy.ndimage import convolve
from PIL import Image
from numpy.lib.stride_tricks import as_strided

def BlockValue(blockData):
    Max1=npmax(npsum(blockData[1:6,1:6],0)) # Inner rows and columns added rowwise
    Min1=npmin(npsum(blockData[1:6,(0,7)],0)) # First and last columns, inner rows, added rowwise
    Max2=npmax(npsum(blockData[1:6,1:6],1)) # Inner rows and columns added columnwise
    Min2=npmin(npsum(blockData[(0,7),1:6],1)) # First and last rows, inner colums, added columnwise
    
    b=Max1-Min1+Max2-Min2
    
    return b

def block_view(A, block=(8, 8)):
    """Provide a 2D block view to 2D array. No error checking made.
    Therefore meaningful (as implemented) only for blocks strictly
    compatible with the shape of A."""
    shape= (int(A.shape[0]/ block[0]), int(A.shape[1]/ block[1]))+ block
    strides= (block[0]* A.strides[0], block[1]* A.strides[1])+ A.strides
    return as_strided(A, shape= shape, strides= strides)

def segmented_stride(M, fun, blk_size=(8,8), overlap=(0,0)):
    # This is some complex function of blk_size and M.shape
    stride = blk_size
    output = np.zeros(M.shape)

    B = block_view(M, block=blk_size)
    O = block_view(output, block=blk_size)

    for b,o in zip(B, O):
        o[:,:] = fun(b);

    return output

def BLK(Filename):
    ImIn=np.array(Image.open(Filename), dtype=np.double)

    YCbCr=rgb2ycbcr(ImIn)
    Y=YCbCr[:,:,1]
    
    # This thresh is used to remove extremely strong edges:
    # block edges are definitely going to be weak
    DiffThresh=50
    
    #Accumulator size. Larger may overcome small splices, smaller may not
    #aggregate enough.
    AC=33
   
    YH=np.insert(Y,0,Y[0,:],axis=0)
    YH=np.append(YH,[Y[-1,:]],axis=0)
    Im2DiffY=-np.diff(YH,2,0)
    Im2DiffY[np.abs(Im2DiffY)>DiffThresh]=0

    padsize=np.round((AC-1)/2).astype(int)
    padded=np.pad(Im2DiffY,((0, 0),(padsize,padsize)),mode='symmetric')

    summedH=convolve(np.abs(padded),np.ones((1,AC)))
    summedH = summedH[:,padsize+1:1-padsize]
    
    mid=medfilt(summedH,[AC,1])
    eH=summedH-mid
   
    paddedHorz=np.pad(eH,((16,16),(0,0)),mode='symmetric')
    HMx=paddedHorz.shape[0]-32
    HMy=paddedHorz.shape[1]
    HorzMid=np.zeros((HMx,HMy,5))
    HorzMid[:,:,0]=paddedHorz[0:HMx,:]
    HorzMid[:,:,1]=paddedHorz[8:HMx+8,:]
    HorzMid[:,:,2]=paddedHorz[16:HMx+16,:]
    HorzMid[:,:,3]=paddedHorz[24:HMx+24,:]
    HorzMid[:,:,4]=paddedHorz[32:HMx+32,:]
    
    HorzMid=np.median(HorzMid,2)
    
    YV=np.insert(Y,0,Y[:,0],axis=1)
    YV=np.insert(YV,-1,Y[:,-1],axis=1)
    Im2DiffX=-np.diff(YV,2,1)
    Im2DiffX[np.abs(Im2DiffX)>DiffThresh]=0
    
    padded=np.pad(Im2DiffX,((padsize,padsize),(0,0)),mode='symmetric')
    
    summedV=convolve(np.abs(padded),np.ones((AC,1)))
    summedV = summedV[padsize+1:1-padsize,:]

    mid=medfilt(summedV,[1,AC])
    eV=summedV-mid

    paddedVert=np.pad(eV,((0,0),(padsize,padsize)),mode='symmetric')
    VMx=paddedVert.shape[0]
    VMy=paddedVert.shape[1]-32
    VertMid=np.zeros((VMx,VMy,5))
    
    VertMid[:,:,0]=paddedVert[:,0:VMy]
    VertMid[:,:,1]=paddedVert[:,8:VMy+8]
    VertMid[:,:,2]=paddedVert[:,16:VMy+16]
    VertMid[:,:,3]=paddedVert[:,24:VMy+24]
    VertMid[:,:,4]=paddedVert[:,32:VMy+32]
    VertMid=np.median(VertMid,2)
    
    
    BlockDiff=HorzMid+VertMid
    
    b=segmented_stride(BlockDiff,BlockValue, (8,8)) #'PadPartialBlocks',1)

    return [b,eH,HorzMid,eV,VertMid,BlockDiff]

