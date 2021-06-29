#from rnd2mtx import rnd2mtx
import numpy as np
#from block_avg import block_avg
from scipy.signal import convolve2d

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
