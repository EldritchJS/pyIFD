"""
This module provides the CFA1 Algorithm

Color-filter-array-artifact-based detector, solution 1.

Algorithm attribution:
P. Ferrara, T. Bianchi, A. De Rosa and P. Piva,
"Image Forgery Localization via Fine-Grained Analysis of CFA Artifacts",
IEEE Transactions on Information Forensics & Security, vol. 7,  no. 5,
 Oct. 2012 (published online June 2012),  pp. 1566-1577.

Based on code from:
Zampoglou, M., Papadopoulos, S., & Kompatsiaris, Y. (2017). Large-scale evaluation of splicing localization algorithms for web images. Multimedia Tools and Applications, 76(4), 4801–4834.
"""

import cv2
import cupy as cp
from pyIFD.CFA2 import CFATamperDetection_F1
def CFA1(impath):
    """
    Main driver of CFA1
    
    Args:
        impath: path to image
        
    Returns:
        OutputMap: CFA1 main output
    """
    bgr = cv2.imread(impath)
    ImageIn = cp.double(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    toCrop = cp.mod(cp.shape(ImageIn), 2)
    if toCrop[0] != 0:
        ImageIn = ImageIn[:-toCrop[0],:,:]
    if toCrop[1] != 0:
        ImageIn = ImageIn[:,:-toCrop[1],:]
    bayer = CFATamperDetection_F1(ImageIn)[1]
    OutputMap = CFAloc(ImageIn, bayer)
    return OutputMap


from cupy.ma import masked_array as ma
def Feature(sigma, pattern):
    with cp.errstate(divide='ignore', invalid='ignore'):
        return cp.prod(ma(sigma,(1-pattern)))/cp.prod(ma(sigma,pattern))


def ApplyFunction(M, pattern, blk_size=(8, 8)):
    """
    Applies BlockValue function to blocks of icput

    Args:
        M: 2d array.
        blk_size (optional, default=(8,8)):

    Returns:
        OutputMap:
    """
    Blocks = GetBlockView(M, block=blk_size)
    OutputMap = cp.empty((int(cp.ceil(cp.shape(M)[0]/blk_size[0])), int(cp.ceil(cp.shape(M)[1]/blk_size[1]))))
    OutputMap[:] = cp.NaN
    for x in range(Blocks.shape[0]):
        for y in range(Blocks.shape[1]):
            OutputMap[x, y] = Feature(Blocks[x, y], pattern)
    return OutputMap


from scipy.ndimage import convolve
import math
from cupy.ma import masked_invalid
from scipy.ndimage import median_filter
def prediction(im):
    """
    Predictor with a bilinear kernel.
    """
    Hpred = cp.array([[0, 0.25, 0], [0.25, -1, 0.25], [0, 0.25, 0]], dtype="double")
    pred_error = convolve(cp.double(im), Hpred, mode='nearest')
    return pred_error
   

def getVarianceMap(im, Bayer, dim):

    # extend pattern over all image
    pattern = cp.kron(cp.ones((int(dim[0]/2),int(dim[1]/2))), Bayer)

    # separate acquired and interpolate pixels for a 7x7 window
    mask = cp.array([[1, 0, 1, 0, 1, 0, 1],
                    [0, 1, 0, 1, 0, 1, 0],
                    [1, 0, 1, 0, 1, 0, 1],
                    [0, 1, 0, 1, 0, 1, 0],
                    [1, 0, 1, 0, 1, 0, 1],
                    [0, 1, 0, 1, 0, 1, 0],
                    [1, 0, 1, 0, 1, 0, 1]])

    # gaussian window for mean and variance
    N_window = 7
    sigma = 1   
    sigrange = cp.arange(-cp.ceil(sigma*2),cp.ceil(sigma*2)+4*sigma/(N_window-1),4*sigma/(N_window-1))
    x = cp.tile(sigrange, (len(sigrange), 1))
    y = cp.tile(sigrange[:, None], (1, len(sigrange)))
    gaussian_window = (1/(2*math.pi*sigma**2))*cp.exp(-0.5*(x**2+y**2)/sigma**2)
    window = gaussian_window*mask
    mc = cp.sum(window)
    vc = 1 - (cp.sum(window**2))
    window_mean = window/mc
    # local variance of acquired pixels
    acquired = im*pattern
    mean_map_acquired = convolve(acquired, window_mean, mode='nearest')*pattern
    sqmean_map_acquired = convolve(acquired**2, window_mean, mode='nearest')*pattern
    var_map_acquired = (sqmean_map_acquired - (mean_map_acquired**2))/vc
    # local variance of interpolated pixels
    interpolated = im*(1-pattern)
    mean_map_interpolated = convolve(interpolated, window_mean, mode='nearest')*(1-pattern)
    sqmean_map_interpolated = convolve(interpolated**2, window_mean, mode='nearest')*(1-pattern)
    var_map_interpolated = (sqmean_map_interpolated - (mean_map_interpolated**2))/vc

    var_map = var_map_acquired + var_map_interpolated

    return var_map


def getFeature(inmap, Bayer, Nb):

    # Proposed feature to localize CFA artifacts

    pattern = cp.kron(cp.ones((int(Nb/2), int(Nb/2))), Bayer)
    statistics = ApplyFunction(inmap, pattern, (Nb, Nb))
    statistics[cp.isnan(statistics)] = 1
    statistics[cp.isinf(statistics)] = 0

    return statistics


def EMGaussianZM(x, tol, max_iter):
    #
    # estimate Gaussian mixture parameters from data x with EM algorithm
    # assume x distributed as alpha * N(0,v1) + (1 - alpha) * N(mu2, v2)

    # initial guess
    alpha = 0.5
    mu2 = cp.mean(x)
    v2 = cp.var(x)
    v1 = v2/10


    alpha_old = 1
    k = 1
    while abs(alpha - alpha_old) > tol and k < max_iter:
        alpha_old = alpha
        k += 1
        # expectation
        f1 = alpha * cp.exp(-x**2/2/v1)/math.sqrt(v1)
        f2 = (1 - alpha) * cp.exp(-(x - mu2)**2/2/v2)/math.sqrt(v2)
        alpha1 = f1 / (f1 + f2)
        alpha2 = f2 / (f1 + f2)
        # maximization
        alpha = cp.mean(alpha1)
        v1 = cp.sum(alpha1 * x**2) / cp.sum(alpha1)
        mu2 = cp.sum(alpha2 * x) / cp.sum(alpha2)
        v2 = cp.sum(alpha2 * (x - mu2)**2) / cp.sum(alpha2)

    # if abs(alpha - alpha_old) > tol:
    #    display('warning: EM algorithm: number of iterations > max_iter');
    

    return [alpha, v1, mu2, v2]


def MoGEstimationZM(statistics):
    # Expectation Maximization Algorithm with Zero-Mean forced first component 

    # E/M algorithm parameters inizialization

    tol = 1e-3
    max_iter = 500

    # NaN and Inf management

    statistics[cp.isnan(statistics)] = 1
    statistics[statistics < 0] = 0
    with cp.errstate(divide='ignore'):
        data = cp.log(cp.ndarray.flatten(statistics), order='F')
    data=masked_invalid(data).compressed()

    # E/M algorithm

    [alpha, v1, mu2, v2] = EMGaussianZM(data, tol, max_iter)

    # Estimated model parameters
    mu = [mu2, 0]

    sigma = cp.sqrt([v2, v1])
    
    return [mu, sigma]


def loglikelihood(statistics, mu, sigma):

    # Loglikelihood map

    # allowable values for logarithm
    min = 1e-320
    max = 1e304

    statistics[statistics == 0] = min
    statistics[statistics < 0] = 0

    mu1=mu[1]
    mu2=mu[0]

    sigma1=sigma[1]
    sigma2=sigma[0]
    # log likelihood
    logstat=cp.log(statistics)
    LogLikelihood = math.log(sigma1) - math.log(sigma2) -0.5*((((logstat - mu2)**2)/sigma2**2) - (((logstat - mu1)**2)/sigma1**2))

    return LogLikelihood


def CFAloc(image, Bayer, Nb=8, Ns=1):
    # parameters
    Nm = 5  # dimension of map filtering
    
    # green channel extraction
    
    im = image[:, :, 1]
    
    [h, w] = cp.shape(im)
    dim = [h, w]
    
    # prediction error
    pred_error = prediction(im)
    
    # local variance of acquired and interpolated pixels
    var_map = getVarianceMap(pred_error, Bayer, dim)
        
    # proposed feature
    stat = getFeature(var_map, Bayer, Nb)
    # GMM parameters estimation
    [mu, sigma] = MoGEstimationZM(stat)
    if sigma[0] == 0 or sigma[1] == 0:
        return cp.zeros(cp.shape(stat), dtype='uint8')
    
    # likelihood map
    loglikelihood_map = loglikelihood(stat, mu, sigma)
    # filtered and cumulated log-likelihood map
    mapLog = median_filter(loglikelihood_map, [Nm, Nm])
    with cp.errstate(over='ignore'):
        expMap = cp.exp(mapLog)
    probMap = 1/(expMap+1)
    
    return probMap



from cupy.lib.stride_tricks import as_strided as ast
def GetBlockView(A, block=(8, 8)):
    shape= (int(cp.floor(A.shape[0]/ block[0])), int(cp.floor(A.shape[1]/ block[1])))+ block
    strides= (block[0]* A.strides[0], block[1]* A.strides[1])+ A.strides
    return ast(A, shape=shape, strides=strides)
