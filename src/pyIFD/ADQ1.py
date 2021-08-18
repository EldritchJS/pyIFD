"""
This module provides the ADQ1 module.

Aligned-double-JPEG-compression-based detector, solution 1.

Algorithm attribution:
Lin, Zhouchen, Junfeng He, Xiaoou Tang, and Chi-Keung Tang. "Fast, automatic
and fine-grained tampered JPEG image detection via DCT coefficient analysis."
Pattern Recognition 42, no. 11 (2009): 2492-2501.


Based on code from:
Zampoglou, M., Papadopoulos, S., & Kompatsiaris, Y. (2017). Large-scale evaluation of splicing localization algorithms for web images. Multimedia Tools and Applications, 76(4), 4801–4834.
"""

import numpy as np
import jpegio as jio
from scipy.signal import medfilt2d
from pyIFD.util import bdct
import matplotlib.image as mpimg


def ExtractYDCT(im):
    """
    Determines YDCT.

    Args:
        im:

    Returns:
        YDCT:
    """
    im = np.double(im)

    Y = 0.299*im[:, :, 0]+0.587*im[:, :, 1]+0.114*im[:, :, 2]
    Y = Y[:int(np.floor(np.shape(Y)[0]/8)*8), :int(np.floor(np.shape(Y)[1]/8)*8)]
    Y -= 128
    YDCT = np.round(bdct(Y, 8)).astype("int")
    return YDCT


def detectDQ_JPEG(im):
    """
    Determines DQ for JPEG image.

    Args:
        im: Input image as read in by JPEGIO

    Returns:
        OutputMap: Heatmap values for detected areas
    """
    # How many DCT coeffs to take into account
    MaxCoeffs = 15
    # JPEG zig-zag sequence
    coeff = [1, 9, 2, 3, 10, 17, 25, 18, 11, 4, 5, 12, 19, 26, 33, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 57, 50, 43, 36, 29, 22,
             15, 8, 16, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 24, 32, 39, 46, 53, 60, 61, 54, 47, 40, 48, 55, 62, 63, 56, 64]

    # Which channel to take: always keep Y only
    channel = 1
    coeffArray = im.coef_arrays[channel-1]
    if im.image_height % 8 != 0:
        coeffArray = coeffArray[:-8, :]
    if im.image_width % 8 != 0:
        coeffArray = coeffArray[:, :-8]
    FFT_Out = {}
    FFT_smoothed = {}
    p_h_fft = np.zeros((MaxCoeffs, 1))
    p_final = np.zeros((MaxCoeffs, 1))
    s_0_Out = np.zeros((MaxCoeffs, 1))
    P_tampered = np.zeros((int(np.shape(coeffArray)[0]/8), int(np.shape(coeffArray)[1]/8), MaxCoeffs))
    P_untampered = np.zeros(np.shape(P_tampered))
    # numOverall = np.zeros(np.shape(P_tampered))
    # denomOverall = np.zeros(np.shape(P_tampered))
    for coeffIndex in range(MaxCoeffs):
        coe = coeff[coeffIndex]
        startY = int(coe % 8)
        if startY == 0:
            startY = 8
        startX = int(np.ceil(coe/8))
        selectedCoeffs = coeffArray[startX-1::8, startY-1::8]
        coeffList = np.reshape(selectedCoeffs, (np.size(selectedCoeffs), 1), order='F')
        minHistValue = int(min(coeffList)-1)
        maxHistValue = int(max(coeffList)+2)
        coeffHist = np.histogram(coeffList, list(range(minHistValue, maxHistValue+1)))[0]
        if(np.size(coeffHist) > 0):
            s_0_Out[coeffIndex] = np.argmax(coeffHist)+1
        # Good through coeffHist
        # Find period by max peak in the FFT minus DC term
        FFT = abs(np.fft.fft(coeffHist))
        FFT_Out[coeffIndex] = FFT
        if np.size(FFT) != 0:
            DC = FFT[0]

            # Find first local minimum, to remove DC peak
            FreqValley = 1
            while (FreqValley < len(FFT)-1) and (FFT[FreqValley-1] >= FFT[FreqValley]):
                FreqValley += 1
            FFT = FFT[FreqValley-1:int(np.floor(len(FFT)/2))]
            FFT_smoothed[coeffIndex] = FFT
            if(np.size(FFT) != 0):
                FFTPeak = np.argmax(FFT)+1
                maxPeak = FFT[FFTPeak-1]
                FFTPeak += FreqValley-1-1  # -1 because FreqValley appears twice, and -1 for the 0-freq DC term
            if np.size(FFT) == 0 or maxPeak < DC/5 or min(FFT)/maxPeak > 0.9:  # threshold at 1/5 the DC and 90% the remaining lowest to only retain significant peaks
                p_h_fft[coeffIndex] = 1
            else:
                p_h_fft[coeffIndex] = round(len(coeffHist)/FFTPeak)
        else:
            p_h_fft[coeffIndex] = 1

        # period is the minimum of the two methods

        p_final[coeffIndex] = p_h_fft[coeffIndex]

        # calculate per-block probabilities
        if p_final[coeffIndex] != 1:
            adjustedCoeffs = selectedCoeffs-minHistValue+1
            period_start = adjustedCoeffs-(np.fmod(adjustedCoeffs-s_0_Out[coeffIndex], p_final[coeffIndex]))
            num = np.zeros(np.shape(period_start))
            denom = np.zeros(np.shape(period_start))
            for kk in range(np.shape(period_start)[0]):
                for ll in range(np.shape(period_start)[1]):
                    if period_start[kk, ll] >= s_0_Out[coeffIndex]:
                        period = list(range(int(period_start[kk, ll]), int(period_start[kk, ll]+p_final[coeffIndex])))
                        if period_start[kk, ll]+p_final[coeffIndex]-1 > len(coeffHist):
                            idx = [i for i, x in enumerate(period) if x > len(coeffHist)]
                            for i in idx:
                                period[i] -= p_final[coeffIndex]
                        num[kk, ll] = coeffHist[adjustedCoeffs[kk, ll]-1]
                        denom[kk, ll] = sum([coeffHist[int(p-1)] for p in period])
                    else:
                        period = list(range(int(period_start[kk, ll]), int(period_start[kk, ll]-p_final[coeffIndex]), -1))
                        if period_start[kk, ll]-p_final[coeffIndex]+1 <= 0:
                            idx = [i for i, x in enumerate(period) if x <= 0]
                            for i in idx:
                                period[i] += p_final[coeffIndex]
                        num[kk, ll] = coeffHist[adjustedCoeffs[kk, ll]-1]
                        denom[kk, ll] = sum([coeffHist[int(p - 1)] for p in period])

            P_u = num/denom
            P_t = 1/p_final[coeffIndex]

            P_tampered[:, :, coeffIndex] = P_t/(P_u+P_t)
            P_untampered[:, :, coeffIndex] = P_u/(P_u+P_t)
        else:
            P_tampered[:, :, coeffIndex] = np.ones((int(np.ceil(np.shape(coeffArray)[0]/8)), int(np.ceil(np.shape(coeffArray)[1]/8))))*0.5
            P_untampered[:, :, coeffIndex] = 1-P_tampered[:, :, coeffIndex]
    P_tampered_Overall = np.prod(P_tampered, axis=2)/(np.prod(P_tampered, axis=2)+np.prod(P_untampered, axis=2))
    P_tampered_Overall[np.isnan(P_tampered_Overall)] = 0

    OutputMap = P_tampered_Overall.copy()

    s = np.var(np.reshape(P_tampered_Overall, (np.size(P_tampered_Overall), 1)))
    Teval = np.zeros((99, 1))
    for S in range(1, 100):
        T = S/100
        Class0 = P_tampered_Overall < T
        Class1 = P_tampered_Overall >= T
        if np.all(Class0) is False:
            s0 = 0
        else:
            s0 = np.var(P_tampered_Overall[Class0])
        if np.all(Class1) is False:
            s1 = 0
        else:
            s1 = np.var(P_tampered_Overall[Class1])
        if s0 == 0 and s1 == 0:
            Teval[S-1] = 0
        else:
            Teval[S-1] = s/(s0+s1)

    Topt = np.argmax(Teval)+1
    Topt = Topt/100-0.01

    Class0 = P_tampered_Overall < Topt
    Class1 = P_tampered_Overall >= Topt

    if np.all(Class0) is False:
        s0 = 0
    else:
        s0 = np.var(P_tampered_Overall[Class0])
    if np.all(Class1) is False:
        s1 = 0
    else:
        s1 = np.var(P_tampered_Overall[Class1])

    Class1_filt = medfilt2d(np.array(Class1, dtype="uint8"), [3, 3])
    Class0_filt = medfilt2d(np.array(Class0, dtype="uint8"), [3, 3])
    e_i = (Class0_filt[:-2, 1:-1]+Class0_filt[1:-1, :-2]+Class0_filt[2:, 1:-1]+Class0_filt[1:-1, 2:])*Class1_filt[1:-1, 1:-1]
    e_i = e_i.astype("double")
    if np.sum(Class0) > 0 and np.sum(Class0) < np.size(Class0):
        K_0 = np.sum(np.maximum(e_i-2, 0))/np.sum(Class0)
    else:
        K_0 = 1
        s0 = 0
        s1 = 0
    Feature_Vector = [Topt, s, s0+s1, K_0]
    return [OutputMap, Feature_Vector, coeffArray]


def detectDQ_NonJPEG(im):
    """
    Determines DQ for non-JPEG.

    Args:
        im:

    Returns:
        OutputMap: Heatmap values for detected areas
    """
    # How many DCT coeffs to take into account
    MaxCoeffs = 15
    # JPEG zig-zag sequence
    coeff = [1, 9, 2, 3, 10, 17, 25, 18, 11, 4, 5, 12, 19, 26, 33, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 57, 50, 43, 36, 29, 22,
             15, 8, 16, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 24, 32, 39, 46, 53, 60, 61, 54, 47, 40, 48, 55, 62, 63, 56, 64]

    # Which channel to take: always keep Y only
    # channel = 1
    coeffArray = ExtractYDCT(im)
    FFT_Out = {}
    FFT_smoothed = {}
    p_h_fft = np.zeros((MaxCoeffs, 1))
    p_final = np.zeros((MaxCoeffs, 1))
    s_0_Out = np.zeros((MaxCoeffs, 1))
    P_tampered = np.zeros((int(np.shape(coeffArray)[0]/8), int(np.shape(coeffArray)[1]/8), MaxCoeffs))
    P_untampered = np.zeros(np.shape(P_tampered))
    # numOverall = np.zeros(np.shape(P_tampered))
    # denomOverall = np.zeros(np.shape(P_tampered))
    for coeffIndex in range(MaxCoeffs):
        coe = coeff[coeffIndex]
        startY = int(coe % 8)
        if startY == 0:
            startY = 8
        startX = int(np.ceil(coe/8))
        selectedCoeffs = coeffArray[startX-1::8, startY-1::8]
        coeffList = np.reshape(selectedCoeffs, (np.size(selectedCoeffs), 1), order='F')
        minHistValue = int(min(coeffList)-1)
        maxHistValue = int(max(coeffList)+2)
        coeffHist = np.histogram(coeffList, list(range(minHistValue, maxHistValue+1)))[0]
        if(np.size(coeffHist) > 0):
            s_0_Out[coeffIndex] = np.argmax(coeffHist)+1
        # Good through coeffHist
        # Find period by max peak in the FFT minus DC term
        FFT = abs(np.fft.fft(coeffHist))
        FFT_Out[coeffIndex] = FFT
        if np.size(FFT) != 0:
            DC = FFT[0]

            # Find first local minimum, to remove DC peak
            FreqValley = 1
            while (FreqValley < len(FFT)-1) and (FFT[FreqValley-1] >= FFT[FreqValley]):
                FreqValley += 1
            FFT = FFT[FreqValley-1:int(np.floor(len(FFT)/2))]
            FFT_smoothed[coeffIndex] = FFT
            FFTPeak = np.argmax(FFT)+1
            maxPeak = FFT[FFTPeak-1]
            FFTPeak += FreqValley-1-1  # -1 because FreqValley appears twice, and -1 for the 0-freq DC term
            if maxPeak < DC/5 or min(FFT)/maxPeak > 0.9:  # threshold at 1/5 the DC and 90% the remaining lowest to only retain significant peaks
                p_h_fft[coeffIndex] = 1
            else:
                p_h_fft[coeffIndex] = round(len(coeffHist)/FFTPeak)
        else:
            p_h_fft[coeffIndex] = 1

        # period is the minimum of the two methods

        p_final[coeffIndex] = p_h_fft[coeffIndex]

        # calculate per-block probabilities
        if p_final[coeffIndex] != 1:
            adjustedCoeffs = selectedCoeffs-minHistValue+1
            period_start = adjustedCoeffs-(np.fmod(adjustedCoeffs-s_0_Out[coeffIndex], p_final[coeffIndex]))
            num = np.zeros(np.shape(period_start))
            denom = np.zeros(np.shape(period_start))
            for kk in range(np.shape(period_start)[0]):
                for ll in range(np.shape(period_start)[1]):
                    if period_start[kk, ll] >= s_0_Out[coeffIndex]:
                        period = list(range(int(period_start[kk, ll]), int(period_start[kk, ll]+p_final[coeffIndex])))
                        if period_start[kk, ll]+p_final[coeffIndex]-1 > len(coeffHist):
                            idx = [i for i, x in enumerate(period) if x > len(coeffHist)]
                            for i in idx:
                                period[i] -= p_final[coeffIndex]
                        num[kk, ll] = coeffHist[adjustedCoeffs[kk, ll]-1]
                        denom[kk, ll] = sum([coeffHist[int(p-1)] for p in period])
                    else:
                        period = list(range(int(period_start[kk, ll]), int(period_start[kk, ll]-p_final[coeffIndex]), -1))
                        if period_start[kk, ll]-p_final[coeffIndex]+1 <= 0:
                            idx = [i for i, x in enumerate(period) if x <= 0]
                            for i in idx:
                                period[i] += p_final[coeffIndex]
                        num[kk, ll] = coeffHist[adjustedCoeffs[kk, ll]-1]
                        denom[kk, ll] = sum([coeffHist[int(p-1)] for p in period])

            P_u = num/denom
            P_t = 1/p_final[coeffIndex]

            P_tampered[:, :, coeffIndex] = P_t/(P_u+P_t)
            P_untampered[:, :, coeffIndex] = P_u/(P_u+P_t)
        else:
            P_tampered[:, :, coeffIndex] = np.ones((int(np.ceil(np.shape(coeffArray)[0]/8)), int(np.ceil(np.shape(coeffArray)[1]/8))))*0.5
            P_untampered[:, :, coeffIndex] = 1-P_tampered[:, :, coeffIndex]
    P_tampered_Overall = np.prod(P_tampered, axis=2)/(np.prod(P_tampered, axis=2)+np.prod(P_untampered, axis=2))
    P_tampered_Overall[np.isnan(P_tampered_Overall)] = 0

    OutputMap = P_tampered_Overall.copy()

    s = np.var(np.reshape(P_tampered_Overall, (np.size(P_tampered_Overall), 1)))
    Teval = np.zeros((99, 1))
    for S in range(1, 100):
        T = S/100
        Class0 = P_tampered_Overall < T
        Class1 = P_tampered_Overall >= T
        s0 = np.var(P_tampered_Overall[Class0])
        s1 = np.var(P_tampered_Overall[Class1])
        Teval[S-1] = s/(s0+s1)

    Topt = np.argmax(Teval)+1
    Topt = Topt/100-0.01

    Class0 = P_tampered_Overall < Topt
    Class1 = P_tampered_Overall >= Topt

    s0 = np.var(P_tampered_Overall[Class0])
    s1 = np.var(P_tampered_Overall[Class1])

    Class1_filt = medfilt2d(np.array(Class1, dtype="uint8"), [3, 3])
    Class0_filt = medfilt2d(np.array(Class0, dtype="uint8"), [3, 3])
    e_i = (Class0_filt[:-2, 1:-1]+Class0_filt[1:-1, :-2]+Class0_filt[2:, 1:-1]+Class0_filt[1:-1, 2:])*Class1_filt[1:-1, 1:-1]
    e_i = e_i.astype("double")
    return OutputMap


def detectDQ(impath):
    """
    Main driver for ADQ1 algorithm

    Args:
        impath: Input image path

    Returns:
        OutputMap: Heatmap values for detected areas
    """
    if impath[-4:] == ".jpg":
        try:
            OutputMap = detectDQ_JPEG(jio.read(impath))
        except Exception as e:
            print('JPEGIO exception: ' + str(e))
            return
    else:
           im = mpimg.imread(impath)
           im = np.round(im*255)
           OutputMap = detectDQ_NonJPEG(im)
    return OutputMap
