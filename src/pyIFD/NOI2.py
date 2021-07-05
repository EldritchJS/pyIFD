import numpy as np
import numpy.matlib
from skimage.color import rgb2ycbcr
from PIL import Image
import PIL
from scipy.signal import convolve2d

def conv2(x, y, mode='same'):
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)

def GetNoiseMaps_hdd( image, filter_type, filter_size, block_rad ):
    # Markos Zampoglou: This a variant version of the code, which calls
    # localNoiVarEstimate_hdd, a version in which intermediate data are
    # stored on disk
    
    YCbCr=np.double(rgb2ycbcr(image))
    im=np.round(YCbCr[:,:,0])
    
    flt = np.ones((filter_size,1))
    flt = (flt*np.transpose(flt))/(filter_size**2)
    noiIm = conv2(im,flt,'same') 

    estV_tmp = localNoiVarEstimate_hdd(noiIm, filter_type, filter_size, block_rad)
    estV = np.array(Image.fromarray(estV_tmp).resize(np.flip(np.round(np.asarray(np.shape(estV_tmp))/4)).astype(int),resample=PIL.Image.BOX))     
    estV[estV<=0.001]=np.mean(estV)
    return estV


def GetNoiseMaps_ram( im, filter_type, filter_size, block_rad ):
    # Markos Zampoglou: This is the original version of the code, where all
    # processing takes place in memory
    
    YCbCr=np.double(rgb2ycbcr(im))
    im=np.round(YCbCr[:,:,0])
    
    flt = np.ones((filter_size,1))
    flt = (flt*np.transpose(flt))/(filter_size**2)
    noiIm = conv2(im,flt,'same')
    
    #estV_tmp = localNoiVarEstimate_ram(noiIm, filter_type, filter_size, block_rad)
    #estV = imresize(single(estV_tmp),round(size(estV_tmp)/4),'method','box')
    estV_tmp = localNoiVarEstimate_hdd(noiIm, filter_type, filter_size, block_rad)
    
    estV = imresize(single(estV_tmp),np.round(np.size(estV_tmp)/4),'method','box')  # TODO: is single necessary?

    estV[estV<=0.001]=np.mean(estV)

    return estV

def block_avg(X,d,pad='zero'): 
# BLOCK_SUM: Compute the avg of elements for all overlapping dxd windows
#            in data X, where d = 2*rad+1.
#
# [bksum] = block_avg(X,rad)
# 
# input arguments:
    #X: an [nx,ny,ns] array as a stack of ns images of size [nx,ny]
    #rad: radius of the sliding window, i.e., window size = (2*rad+1)*(2*rad+1)
#  pad: padding patterns:
            #'zero': padding with zeros (default)
            #'mirror': padding with mirrored boundary area
#
# output arguments:
#    bksum:sum of elements for all overlapping dxd windows
#
# Xunyu Pan, Xing Zhang and Siwei Lyu -- 07/26/2012             

    [nx,ny,ns] = np.shape(X)
    if d < 0 or d != np.floor(d) or d >= min(nx,ny):
    #error('window size needs to be a positive integer');
        return
    wd = 2*d+1 # size of the sliding window

    Y = np.zeros((nx+wd,ny+wd,ns),'single')
    Y[d+1:nx+d+1,d+1:ny+d+1,:] = X 

    # padding boundary
    if pad[0:2] != 'ze':
        if pad[0:2] == 'mi':
    # padding by mirroring
        # mirroring top
            Y[1:d+1,:,:] = np.flip(Y[d+2:wd+1,:,:],axis=0)
            # mirroring bottom
            Y[nx+d+1:,:,:] = np.flip(Y[nx:nx+d,:,:],axis=0)
            # mirroring left
            Y[:,1:d+1,:] = np.flip(Y[:,d+2:wd+1,:],axis=1)
            # mirroring right
            Y[:,ny+d+1:,:] = np.flip(Y[:,ny:ny+d,:],axis=1)
        else:
            #error('unknown padding pattern');
            return
    
# forming integral image
    Y = np.cumsum(np.cumsum(Y,0),1)

    # computing block sums
    Y = Y[wd:,wd:,:]+Y[:-wd,:-wd,:] - Y[wd:,:-wd,:]-Y[:-wd,wd:,:]
    Y /=(wd*wd)
    return Y 

def dct2mtx(n,ord): 
# DCT2MTX: generating matrices corresponding to 2D-DCT transform.
#          
#
# [mtx] = dct2mtx(N)
# 
# input arguments:
#	N: size of 2D-DCT basis (N x N)
#  ord: order of the obtained DCT basis
#		'grid': as grid order (default)
#     'snake': as snake order
# output arguments:
#	mtx: 3D matrices of dimension (NxNxN^2)
#       mtx(:,:,k) is the kth 2D DCT basis of support size N x N
#
# Xunyu Pan, Xing Zhang, Siwei Lyu -- 07/26/2012             
    mtx = np.zeros((n,n,n*n))
    return mtx

def haar2mtx(n): 
# DCT2MTX: generating matrices corresponding to 2D-Haar wavelet transform.
#          
#
# [mtx] = haar2mtx(N)
# 
# input arguments:
#	N: size of 2D-DCT basis (N x N)
#	   has to be power of two
# output arguments:
#	mtx: 3D matrices of dimension (NxNxN^2)
#       mtx(:,:,k) is the kth 2D haar basis of support size N x N
#
# Xunyu Pan, Xing Zhang, Siwei Lyu -- 07/26/2012             
#
# code borrowed from Jin Qi's HaarTransforamtionMatrix function

# check input parameter and make sure it's the power of 2
    Level=np.log2(n)
    if 2**Level<n:
        return
#     error('input parameter has to be the power of 2');

#Initialization
    c=[1]
    NC=1/np.sqrt(2) #normalization constant
    LP=[1,1]
    HP=[1,-1]

    # iteration from H=[1] 
#    for i in range(0,Level):
#        c = NC*[np.kron(c,LP);np.kron(np.eye(np.shape(c)),HP)]


    mtx = np.zeros((n,n,n*n))
#    k = 0
#    for i in range(0,n):
#        for j in range(0,n):
#	        mtx[:,:,k] = np.transpose(c[i,])*c[j,:]
#	        k = k+1
    return mtx

def conv2(x, y, mode='same'):
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)

def localNoiVarEstimate_hdd(noi,ft,fz,br):
    # Markos Zampoglou: this is a variant of the original
    # localNoiVarEstimate.m, aimed to be more memory-efficient. The
    # original has been renamed to localNoiVarEstimate_ram
    #
    # localNoiVarEstimate: local noise variance estimation using kurtosis
    #
    # [estVar] = localNoiVarEstimate(noisyIm,filter_type,filter_size,block_size)
    #
    # input arguments:
    #	noisyIm: input noisy image
    #	filter_type: the type of band-pass filter used
    #        supported types, "dct", "haar", "rand"
    #   filter_size: the size of the support of the filter
    #   block_rad: the size of the local blocks
    # output arguments:
    #	estVar: estimated local noise variance
    #
    # reference:
    #   X.Pan, X.Zhang and S.Lyu, Exposing Image Splicing with
    #   Inconsistent Local Noise Variances, IEEE International
    #   Conference on Computational Photography, Seattle, WA, 2012
    #
    # disclaimer:
    #	Please refer to the ReadMe.txt
    #
    # Xunyu Pan, Xing Zhang and Siwei Lyu -- 07/26/2012
    
    if ft == 'dct':
        fltrs = dct2mtx(fz,'snake')
    elif ft == 'haar':
        fltrs = haar2mtx(fz)
    elif ft == 'rand':
        fltrs = rnd2mtx(fz)
    else:
        return 0
    # decompose into channels
    ch = np.zeros([np.shape(noi)[0],np.shape(noi)[1],fz*fz-1],'single');
    for k in range(1,fz**2):
        ch[:,:,k-1] = conv2(noi,fltrs[:,:,k],'same');
    # collect raw moments
    blksz = (2*br+1)*(2*br+1)
    mu1 = block_avg(ch,br,'mi')
    mu2 = block_avg(ch**2,br,'mi');
    mu3 = block_avg(ch**3,br,'mi');
    mu4 = block_avg(ch**4,br,'mi');
    Factor34=mu4 - 4*mu1*mu3;
    noiV = mu2 - mu1**2
    noiK = (Factor34 + 6*mu1**2*mu2 - 3*mu1**4)/(noiV**2)-3
    noiK[noiK<0]=0


    a = np.mean(np.sqrt(noiK),2)
    b = np.mean(1/noiV,2)
    c = np.mean(1/noiV**2,2)
    d = np.mean(np.sqrt(noiK)/noiV,2)
    e = np.mean(noiV,2);

    sqrtK = (a*c - b*d)/(c-b*b)

    V=np.zeros(a.shape,'single')
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if(sqrtK[i,j]==0):
                V[i,j]=1/b[i,j]
            else:
                V[i,j]=(1-a[i,j]/sqrtK[i,j])/b[i,j]

    idx = sqrtK<np.median(sqrtK)
    V[idx] = 1/b[idx]
    idx = V<0
    V[idx] = 1/b[idx]
    
    return V

def rnd2mtx(n): 
# DCT2MTX: generating matrices corresponding to random orthnormal transform.
#
# [mtx] = rnd2mtx(N)
# 
# input arguments:
#	N: size of 2D random basis (N x N)
#
# output arguments:
#	mtx: 3D matrices of dimension (NxNxN^2)
#       mtx(:,:,k) is the kth 2D DCT basis of support size N x N
#
# Xunyu Pan, Xing Zhang, Siwei Lyu -- 07/26/2012             

    X = np.random.randn(n,n)
    X -= np.matlib.repmat(np.mean(X,0),n,1)
    X /=np.matlib.repmat(np.sqrt(np.sum(X**2,0)),n,1)

    mtx = np.zeros((n,n,n*n))
    k = 0 
    for i in range(n):
        for j in range(n):
            mtx[:,:,k] = np.outer(X[:,i],np.transpose(X[:,j]))
            k+=1
    return mtx

def GetNoiseMaps( filename, sizeThreshold=55*(2**5), filter_type='rand', filter_size=4, block_rad=8 ):
    # Copyright (C) 2016 Markos Zampoglou
    # Information Technologies Institute, Centre for Research and Technology Hellas
    # 6th Km Harilaou-Thermis, Thessaloniki 57001, Greece
    #
    # This code implements the algorithm presented in:
    # Lyu, Siwei, Xunyu Pan, and Xing Zhang. "Exposing region splicing
    # forgeries with blind local noise estimation." International Journal
    # of Computer Vision 110, no. 2 (2014): 202-221. 
    
    # Due to extremely high memory requirements,
    # especially for large images, this function detects large images and
    # runs a memory efficient version of the code, which stores
    # intermediate data to disk (GetNoiseMaps_hdd)
    im = Image.open(filename)
    size=np.prod(np.shape(im))
    if size>sizeThreshold:
        #disp('hdd-based');
        estV = GetNoiseMaps_hdd( im, filter_type, filter_size, block_rad )
    else:
        #disp('ram-based');
        estV = GetNoiseMaps_ram( im, filter_type, filter_size, block_rad )
    
    return estV
