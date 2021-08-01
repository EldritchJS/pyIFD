Module pyIFD.NOI5
=================
This module provides the NOI5 algorithm

Functions
---------

    
`KMeans(data, N)`
:   Sorts data into N bins.
    
    Args:
        data: data to be sorted
        N: number of bins to be sorted into
    
    Returns:
        u: means of the bins
        re: If data is a nx1 vector, this will be a nx2 output. The first column will be the point, and the second will be its bin assignment

    
`PCANoise(impath)`
:   Main driver for NOI5 algorithm.
    
    Args:
        impath: input image path.
    
    Returns:
        Noise_mix2: OutputMap
        bwpp: OutputMap (Quantized)

    
`PCANoiseLevelEstimator(image, Bsize)`
:   Summary please.
    
    Args:
        image: Image to process
        Bsize:
    
    Returns:
        label:
        variance: