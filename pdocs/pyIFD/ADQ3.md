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
        OutputMap: Output of ADQ3 algorithm (2D array).

    
`EstimateJPEGQuality(imIn)`
:   Estimates the quality of JPEG object.
    
    Args:
        imIn: jpegio struct
    
    Returns:
        Quality: 0-100 integer

    
`ExtractFeatures(im, c1, c2, ncomp, digitBinsToKeep)`
:   This function extracts a descriptor feature based on the first-digit distribution of DCT coefficients of an image. It is needed by BenfordDQ.
    
     Args:
         c1: first DCT coefficient to be taken into account, DC term included
         c2: final DCT coefficient to be taken into account, DC term included
         ncomp: component from which to extract the feature (1 corresponds to the Y component)
         digitBinsToKeep: digits for which to keep their frequency
    
    Returns:
         output: Flattened feature vector

    
`svmdecision(Xnew, index)`
:   Uses given index of svm to classify Xnew.
    
    Args:
        Xnew: Array to be classifed
        index: Index of SVM to use to classify
    
    Returns:
        f: 2d array of svm decision output.