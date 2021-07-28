Module pyIFD.ADQ3
=================
This module provides the ADQ3 algorithm

Functions
---------

    
`BenfordDQ(impath)`
:   Main driver for ADQ3 algorithm.
    
    Args:
        impath: Input image path, required to be JPEG with extension .jpg
    
    Returns:
        OutputMap:

    
`EstimateJPEGQuality(imIn)`
:   Estimates the quality of JPEG.
    
    Args:
        imIn:   Image
    
    Returns:        
        Quality: (0-100)

    
`ExtractFeatures(im, c1, c2, ncomp, digitBinsToKeep)`
:   This function extracts a descriptor feature based on the first-digit distribution of DCT coefficients of an image. It is needed by BenfordDQ. 
    
     Args:
         c1: first DCT coefficient to be taken into account, DC term included
         c2: final DCT coefficient to be taken into account, DC term included
         ncomp: component from which to extract the feature (1 corresponds to the Y component)
         digitBinsToKeep: digits for which to keep their frequency
    
    Returns:
        np.ndarray.flatten(HistToKeep):

    
`dequantize(qcoef, qtable)`
:   Dequantizes a coef table given a quant table.
    
    Args:
        qcoef:
        qtable:
    
    Returns:
        coef:

    
`im2vec(im, bsize, padsize=0)`
:   Converts image to a vector.
    
    Args:
        im:
        bsize:
        padsize (optional, default=0):
    
    Returns:
        v:
        rows:
        cols:

    
`svmdecision(Xnew, index)`
:   Uses given index of svm to classify Xnew.
    
    Args:
        Xnew:
        index:
    
    Returns:
        f:

    
`vec2im(v, padsize=[0, 0], bsize=None, rows=None, cols=None)`
:   Converts vector to an image.
    
    Args:
        v:
        padsize (optional, default=[0,0]):
        bsize (optional, default=None):
        rows (optional, default=None):
        cols (optional, default=None):
    
    Returns:
        im: