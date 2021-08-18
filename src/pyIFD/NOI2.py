"""
This module provides the NOI2 algorithm

Noise-variance-inconsistency detector, solution 2.

Algorithm attribution:
Lyu, Siwei, Xunyu Pan, and Xing Zhang. "Exposing region splicing forgeries
with blind local noise estimation." International Journal of Computer Vision
110, no. 2 (2014): 202-221.

Based on code from:
Zampoglou, M., Papadopoulos, S., & Kompatsiaris, Y. (2017). Large-scale evaluation of splicing localization algorithms for web images. Multimedia Tools and Applications, 76(4), 4801–4834.
"""

import numpy as np
import cv2
from PIL import Image
from scipy.signal import convolve2d


def conv2(x, y, mode='same'):
    """
    Computes standard 2d convolution for matrices x and y.

    Args:
        x: 2d matrix.
        y: 2d matrix.
        mode (optional, default='same'):

    Returns:
        computation:

    Todos:
        * Sort out return
    """
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)


def GetNoiseMaps_hdd(im, filter_type, filter_size, block_rad):
    """
    Outputs variance estimates for im. Equivalent to GetNoiseMaps_ram

    Args:
        im: Image to be processed.
        filter_type: Type of filter. Must be one of ('haar','dct','rand')
        filter_size: the size of the support of the filter
        block_rad: the size of the local blocks

    Returns:
        estV: estimated local noise variance
    TODO:
        * Consider removing the ram function path.
    """
    origT = [65.481/255, 128.553/255, 24.966/255]
    Y = origT[0]*im[:, :, 2]+origT[1]*im[:, :, 1]+origT[2]*im[:, :, 0]+16
    im = np.round(Y)

    flt = np.ones((filter_size, 1))
    flt = (flt*np.transpose(flt))/(filter_size**2)
    noiIm = conv2(im, flt, 'same')

    estV_tmp = localNoiVarEstimate_hdd(noiIm, filter_type, filter_size, block_rad)
    estVSize = tuple(np.round((np.array(np.shape(estV_tmp))+0.1)/4))
    estV = np.array(Image.fromarray(estV_tmp).resize(np.flip(estVSize).astype(int), resample=Image.BOX))
    estV[estV <= 0.001] = np.mean(estV)
    return estV


def GetNoiseMaps_ram(im, filter_type, filter_size, block_rad):
    """
    Outputs variance estimates for im.

    Args:
        im: Image to be processed.
        filter_type: Type of filter. Must be one of ('haar','dct','rand')
        filter_size: the size of the support of the filter
        block_rad: the size of the local blocks

    Returns:
        estV: estimated local noise variance
    """
    origT = [65.481/255, 128.553/255, 24.966/255]
    Y = origT[0]*im[:, :, 2]+origT[1]*im[:, :, 1]+origT[2]*im[:, :, 0]+16
    im = np.round(Y)

    flt = np.ones((filter_size, 1))
    flt = (flt*np.transpose(flt))/(filter_size**2)
    noiIm = conv2(im, flt, 'same')

    estV_tmp = localNoiVarEstimate_hdd(noiIm, filter_type, filter_size, block_rad)
    estV = np.imresize(estV_tmp, np.round(np.size(estV_tmp)/4), 'method', 'box')
    estV[estV <= 0.001] = np.mean(estV)

    return estV


def block_avg(X, d, pad='zero'):
    """
    Computes the avg of elements for all overlapping dxd windows in data X, where d = 2*rad+1.

    Args:
        X: an [nx,ny,ns] array as a stack of ns images of size [nx,ny]
        rad: radius of the sliding window, i.e., window size = (2*rad+1)*(2*rad+1)
        pad (optional, default='zero'): padding patterns

    Returns:
        Y: sum of elements for all overlapping dxd windows
    """
    [nx, ny, ns] = np.shape(X)
    if d < 0 or d != np.floor(d) or d >= min(nx, ny):
        return

    wd = 2*d+1  # size of the sliding window

    Y = np.zeros((nx+wd, ny+wd, ns), 'single')
    Y[d+1:nx+d+1, d+1:ny+d+1, :] = X

    # padding boundary
    if pad[0:2] != 'ze':
        # padding by mirroring
        if pad[0:2] == 'mi':
            # mirroring top
            Y[1:d+1, :, :] = np.flip(Y[d+2:wd+1, :, :], axis=0)
            # mirroring bottom
            Y[nx+d+1:, :, :] = np.flip(Y[nx:nx+d, :, :], axis=0)
            # mirroring left
            Y[:, 1:d+1, :] = np.flip(Y[:, d+2:wd+1, :], axis=1)
            # mirroring right
            Y[:, ny+d+1:, :] = np.flip(Y[:, ny:ny+d, :], axis=1)
        else:
            return

    # forming integral image
    Y = np.cumsum(np.cumsum(Y, 0), 1)

    # computing block sums
    Y = Y[wd:, wd:, :]+Y[:-wd, :-wd, :] - Y[wd:, :-wd, :]-Y[:-wd, wd:, :]
    Y /= (wd*wd)
    return Y


def dct2mtx(n, order):
    """
    Generates matrices corresponding to 2D-DCT transform.

    Args:
        N: size of 2D-DCT basis (N x N)
        ord: order of the obtained DCT basis

    Returns:
        mtx: 3D matrices of dimension (NxNxN^2)
    """
    (cc, rr) = np.meshgrid(range(n), range(n))

    c = np.sqrt(2 / n) * np.cos(np.pi * (2*cc + 1) * rr / (2 * n))
    c[0, :] = c[0, :] / np.sqrt(2)
    if order[:2] == 'gr':
        order = np.reshape(range(n**2), (n, n), order='F')
    elif order[:2] == 'sn':  # not exactly snake code,but close
        temp = cc+rr
        idx = np.argsort(np.ndarray.flatten(temp))
        order = np.reshape(idx, (n, n), order='F')

    mtx = np.zeros((n, n, n*n))
    for i in range(n):
        for j in range(n):
            mtx[:, :, order[i, j]] = np.outer(c[i, :], c[j, :])

    return mtx


def haar2mtx(n):
    """
    Generates haar filter of size (n,n,n**2).

    Args:
        n: Positive integer.

    Returns:
        mtx: nxn filter array.
    """
    Level = int(np.log2(n))
    if 2**Level < n:
        print("input parameter has to be the power of 2")
        return

    # Initialization
    c = np.ones((1, 1))
    NC = 1/np.sqrt(2)  # normalization constant
    LP = [1, 1]
    HP = [1, -1]

    # iteration from H=[1]
    for i in range(0, Level):
        c = NC*np.concatenate((np.kron(c, LP), np.kron(np.eye(np.shape(c)[0], np.shape(c)[1]), HP)))

    mtx = np.zeros((n, n, n*n))
    k = 0
    for i in range(n):
        for j in range(n):
            mtx[:, :, k] = np.outer(c[i, :], c[j, :])
            k += 1
    return mtx


def localNoiVarEstimate_hdd(noi, ft, fz, br):
    """
    Computes local noise variance estimation using kurtosis.

    Args:
        noisyIm: input noisy image
        filter_type: the type of band-pass filter used supported types, "dct", "haar", "rand"
        filter_size: the size of the support of the filter
        block_rad: the size of the local blocks

    Returns:
        estVar: estimated local noise variance
     """
    if ft == 'dct':
        fltrs = dct2mtx(fz, 'snake')
    elif ft == 'haar':
        fltrs = haar2mtx(fz)
    elif ft == 'rand':
        fltrs = rnd2mtx(fz)
    else:
        return 0
    # decompose into channels
    ch = np.zeros([np.shape(noi)[0], np.shape(noi)[1], fz*fz-1], 'single')
    for k in range(1, fz**2):
        ch[:, :, k-1] = conv2(noi, fltrs[:, :, k], 'same')
    # collect raw moments
    mu1 = block_avg(ch, br, 'mi')
    mu2 = block_avg(ch**2, br, 'mi')
    mu3 = block_avg(ch**3, br, 'mi')
    mu4 = block_avg(ch**4, br, 'mi')
    Factor34 = mu4 - 4*mu1*mu3
    noiV = mu2 - mu1**2
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        noiK = (Factor34 + 6*mu1**2*mu2 - 3*mu1**4)/(noiV**2)-3
        noiK[noiK < 0] = 0
        a = np.mean(np.sqrt(noiK), 2)
        b = np.mean(1/noiV, 2)
        c = np.mean(1/noiV**2, 2)
        d = np.mean(np.sqrt(noiK)/noiV, 2)

        sqrtK = (a*c - b*d)/(c-b*b)
        V = (1-a/sqrtK)/b
        V = V.astype("single")

        idx = sqrtK < np.median(sqrtK)
        V[idx] = 1/b[idx]
        idx = V < 0
        V[idx] = 1/b[idx]
    return V


def rnd2mtx(n):
    """
     Generates matrices corresponding to random orthnormal transform.

     Args:
        N: size of 2D random basis (N x N)

     Returns:
        mtx: 3D matrices of dimension (NxNxN^2)
    """
    X = np.random.randn(n, n)
    X -= np.tile(np.mean(X, 0), (n, 1))
    X /= np.tile(np.sqrt(np.sum(X**2, 0)), (n, 1))

    mtx = np.zeros((n, n, n*n))
    k = 0
    for i in range(n):
        for j in range(n):
            mtx[:, :, k] = np.outer(X[:, i], np.transpose(X[:, j]))
            k += 1
    return mtx


def GetNoiseMaps(impath, sizeThreshold=55*(2**5), filter_type='rand', filter_size=4, block_rad=8):
    """
    Main driver for NOI2 algorithm.

    Args:
        impath:
        sizeThreshold (optional, default=55*25):
        filter_type (optional, default='rand'):
        filter_size (optional, default=4):
        block_rad (optional, default=8):

    Returns:
        estV: Equivalent to OutputMap

    """
    im = cv2.imread(impath)
    size = np.prod(np.shape(im))
    if size > sizeThreshold:
        estV = GetNoiseMaps_hdd(im, filter_type, filter_size, block_rad)
    else:
        estV = GetNoiseMaps_ram(im, filter_type, filter_size, block_rad)
    estV = np.nan_to_num(estV, posinf=0, neginf=0)
    return estV
