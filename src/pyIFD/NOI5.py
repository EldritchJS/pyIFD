"""
This module provides the NOI5 algorithm

Noise-variance-inconsistency detector, solution 5 (leveraging Principal Component Analysis).

Algorithm attribution:
H. Zeng, Y. Zhan, X. Kang, X. Lin, Image splicing localization using PCA-based
noise level estimation, Multimedia Tools & Applications, 2017.76(4):4783
http://www.escience.cn/people/Zenghui/index.html

Based on code from:
Zampoglou, M., Papadopoulos, S., & Kompatsiaris, Y. (2017). Large-scale evaluation of splicing localization algorithms for web images. Multimedia Tools and Applications, 76(4), 4801–4834.
"""

import cupy as cp
from cupy.linalg import eigh
import cv2
from scipy.ndimage import median_filter as medfilt


def KMeans(data, N):
    """
    Sorts data into N bins.

    Args:
        data: data to be sorted
        N: number of bins to be sorted into

    Returns:
        u: means of the bins
        re: If data is a nx1 vector, this will be a nx2 output. The first column will be the point, and the second will be its bin assignment
    """
    m = data.size
    u = cp.zeros((N, 1))
    Sdata = cp.sort(data)
    u[0] = cp.mean(Sdata[-round(m/4)-1:])
    u[1] = cp.mean(Sdata[:round(m/4)])
    umax = cp.median(Sdata[-round(m/10)-1:])
    data[data > umax] = umax
    for iter in range(200):
        pre_u = u.copy()     # center of the last iter
        tmp = cp.zeros((N, m))
        for i in range(N):
            tmp[i, :] = data-u[i]
        tmp = cp.abs(tmp)
        junk = cp.min(tmp, axis=0)
        index = cp.argmin(tmp, axis=0)
        quan = cp.zeros((m, N))
        for i in range(m):
            quan[i, index[i]] = junk[i]
        for i in range(N):
            if (cp.sum(quan[:, i]) > 0.01):
                u[i] = cp.sum(quan[:, i]*data)/cp.sum(quan[:, i])

        if (cp.linalg.norm(pre_u-u) < 0.02):
            break

    re = cp.zeros((m, 2))
    for i in range(m):
        tmp = cp.zeros((N, 1))
        for j in range(N):
            tmp[j] = cp.linalg.norm(data[i]-u[j])

        junk = cp.min(tmp, axis=0)
        index = cp.argmin(tmp, axis=0)
        re[i, 0] = data[i]
        re[i, 1] = index+1
    # the tampered area is less than half of the whole image
    label = re[:, 1]
    if list(label).count(1) < int(m/2):
        re[:, 1] = 3-label

    return [u, re]


def PCANoiseLevelEstimator(image, Bsize):
    """
    Summary please.

    Args:
        image: Image to process
        Bsize:

    Returns:
        label:
        variance:
    """
    UpperBoundLevel = 0.0005
    UpperBoundFactor = 3.1
    M1 = Bsize
    M2 = Bsize
    M = M1 * M2
    EigenValueCount = 7
    EigenValueDiffThreshold = 49.0
    LevelStep = 0.05
    MinLevel = 0.06
    MaxClippedPixelCount = round(cp.nextafter(0.1*M, 0.1*M+1))

    # ==========================================================================
    def Clamp(x, a, b):
        """
        Limit icput value to a range.

        Args:
            x: value to clamp
            a: minimum value
            b: maximum value

        Returns:
            y: clamped value
        """
        y = x
        if x < a:
            y = a
        if x > b:
            y = b
        return y

    # ==========================================================================
    def ComputeBlockInfo( image ):
        """
        Summary please.
        Args:
            image:
        Returns:
            block_info
        """
        sums=cp.zeros((cp.shape(image)[0]-M1,cp.shape(image)[1]))
        block_info = cp.zeros((cp.shape(image)[0]*cp.shape(image)[1],3))
        image2=image**2
        sums2=cp.zeros(cp.shape(sums))
        clipped=cp.zeros(cp.shape(sums))
        for x in range(cp.shape(image)[0]-M2):
            for y in range(cp.shape(image)[1]):

                if x == 0:
                    sums[0,y] = cp.sum(image[:M2,y])
                    sums2[0,y] = cp.sum(image2[:M2,y])
                    clipped[0,y]= cp.count_nonzero((image[:M2,y]==0) | (image[:M2,y]==255))
                else:
                    sums[x,y] = sums[x-1,y]-image[x-1,y]+image[x+M2-1, y]
                    sums2[x, y] = sums2[x-1, y] - image2[x-1,y]+image2[x+M2-1, y]
                    clipped[x, y] = clipped[x-1, y]
                    if image[x-1, y] in [0,255]:
                        clipped[x,y]-=1
                    if image[x+M2-1, y] in [0,255]:
                        clipped[x,y]+=1

        prevsum1=-1
        prevsum2=-1
        prevclipped=-1
        block_count=0
        for y in range(cp.shape(image)[1]-M1):
            for x in range(cp.shape(image)[0]-M2):        
                if x == 0:
                    sum1=cp.sum(sums[y,:M2])
                    sum2=cp.sum(sums2[y,:M2])
                    clipped_pixel_count=cp.sum(clipped[y,:M2])
                else:
                    sum1=prevsum1-sums[y,x-1]+sums[y,x+M2-1]
                    sum2=prevsum2-sums2[y,x-1]+sums2[y,x+M2-1]
                    clipped_pixel_count=prevclipped-clipped[y,x-1]+clipped[y,x+M2-1]
                prevsum1=sum1
                prevsum2=sum2
                prevclipped=clipped_pixel_count
                if clipped_pixel_count <= MaxClippedPixelCount:  
                    block_info[block_count,0] = (sum2 - sum1*sum1/M) / M
                    block_info[block_count,1] = x+1
                    block_info[block_count,2] = y+1
                    block_count += 1 
        block_info=cp.delete(block_info,slice(block_count,cp.shape(image)[0]*cp.shape(image)[1]),0)
        return block_info

    # ==========================================================================
    def ComputeStatistics(image, block_info):
        """
        Summary please.

        Args:
            image:
            block_info:

        Returns:
            sum1:
            sum2:
            subset_size:
        """
        loop_iters = len(cp.arange(1, MinLevel, -0.05))
        sum1 = cp.zeros((M, 1, loop_iters))
        sum2 = cp.zeros((M, M, loop_iters))
        subset_size = cp.zeros((loop_iters, 1))
        subset_count = 0
        max_index = cp.shape(block_info)[0]-1
        for p in cp.arange(1, MinLevel, -LevelStep):
            q = 0
            if p - LevelStep > MinLevel:
                q = p - LevelStep

            beg_index = Clamp(round(q*max_index+LevelStep/2) + 1, 1, max_index+1)
            end_index = Clamp(round(p*max_index+LevelStep/2) + 1, 1, max_index+1)
            curr_sum1 = cp.zeros((M, 1))
            curr_sum2 = cp.zeros((M, M))
            for k in range(int(beg_index)-1, int(end_index)-1):
                curr_x = int(block_info[k, 1])
                curr_y = int(block_info[k, 2])
                block = cp.reshape(image[curr_y-1:curr_y+M2-1, curr_x-1:curr_x+M1-1], (M, 1), order='F').astype("double")
                curr_sum1 += block
                curr_sum2 += block * block.T
            subset_count += 1
            sum1[:, :, subset_count-1] = curr_sum1.copy()
            sum2[:, :, subset_count-1] = curr_sum2.copy()
            subset_size[subset_count-1] = end_index - beg_index
        for i in range(len(subset_size)-1, 0, -1):
            sum1[:, :, i-1] += sum1[:, :, i]
            sum2[:, :, i-1] += sum2[:, :, i]
            subset_size[i-1] += subset_size[i]
        return [sum1, sum2, subset_size]

    # ==========================================================================
    def ComputeUpperBound(block_info):
        """
        Summary please.

        Args:
            block_info:

        Returns:
            upper_bound:
        """
        max_index = cp.shape(block_info)[0] - 1
        zero_idx = cp.where(block_info[:, 0] == 0)[0]
        if zero_idx.size == 0:
            nozeroindex = round(UpperBoundLevel*max_index)
        else:
            nozeroindex = min(cp.max(cp.where(block_info[:, 0] == 0)[0])+1, cp.shape(block_info)[0]-1)
        index = Clamp(round(UpperBoundLevel*max_index) + 1, nozeroindex, cp.shape(block_info)[0]-1)
        upper_bound = UpperBoundFactor * block_info[index, 0]
        return upper_bound

    # ==========================================================================
    def ApplyPCA(sum1, sum2, subset_size):
        """
        Summary please.

        Args:
            sum1: Matrix one for PCA
            sum2: Matrix two for PCA
            subset_size: Vector for subset size

        Returns:
            eigh: Eigenvalues.
        """
        meanval = sum1 / subset_size
        cov_matrix = sum2 / subset_size - meanval * cp.transpose(meanval)
        return eigh(cov_matrix)[0]

    # ==========================================================================
    def GetNextEstimate(sum1, sum2, subset_size, prev_estimate, upper_bound):
        """
        Summary please.

        Args:
            sum1:
            sum2:
            subset_size:
            prev_estimate:
            upper_bound:

        Returns:
            variance:
        """
        variance = 0
        for i in range(len(subset_size)):
            eigen_value = ApplyPCA(sum1[:, :, i], sum2[:, :, i], subset_size[i])
            variance = eigen_value[0]
            if variance < 0.00001:
                break
            diff = eigen_value[EigenValueCount-1] - eigen_value[0]
            diff_threshold = EigenValueDiffThreshold * prev_estimate / subset_size[i]**0.5

            if(diff < diff_threshold and variance < upper_bound):
                break
        return variance

    # ==========================================================================

    label = 0
    block_info = ComputeBlockInfo(image)
    if cp.min(cp.shape(block_info)) == 0:
        label = 1
        variance = cp.var(image)
    else:
        idx = cp.lexsort((block_info[:, 2], block_info[:, 0]))
        block_info = cp.asarray([block_info[i, :] for i in idx])
        [sum1, sum2, subset_size] = ComputeStatistics(image, block_info)
        if subset_size[-1] == 0:
            label = 1
            variance = cp.var(image)
        else:
            upper_bound = ComputeUpperBound(block_info)
            prev_variance = 0
            variance = upper_bound
            for iter in range(10):
                if(cp.abs(prev_variance - variance) < 0.00001):
                    break
                prev_variance = variance
                variance = GetNextEstimate(sum1, sum2, subset_size, variance, upper_bound)
            if variance < 0:
                label = 1
                variance = cp.var(image)
    variance = cp.sqrt(variance)
    return [label, variance]


def PCANoise(impath):
    """
    Main driver for NOI5 algorithm.

    Args:
        impath: icput image path.

    Returns:
        Noise_mix2: OutputMap
        bwpp: OutputMap (Quantized)

    """
    B = 64
    imin = cv2.cvtColor(cv2.imread(impath), cv2.COLOR_BGR2GRAY).astype("double")
    [M, N] = cp.shape(imin)
    imin = cp.array(imin[:int(cp.floor(M/B)*B), :int(cp.floor(N/B)*B)])
    [M, N] = cp.shape(imin)
    irange = int(cp.floor(M/B))
    jrange = int(cp.floor(N/B))
    Ib = cp.zeros((irange, jrange))
    Noise_64 = cp.zeros((irange, jrange))

    for i in range(irange):
        for j in range(jrange):
            Ib = imin[i*B:(i+1)*B, j*B:(j+1)*B]
            Noise_64[i, j] = PCANoiseLevelEstimator(Ib, 5)[1]
    [u, re] = KMeans(Noise_64.flatten(order='F'), 2)
    result4 = cp.reshape(re[:, 1], cp.shape(Noise_64), order='F')

    B = 32
    irange = int(cp.floor(M/B))
    jrange = int(cp.floor(N/B))
    label32 = cp.zeros((irange, jrange))
    Noise_32 = cp.zeros((irange, jrange))
    for i in range(irange):
        for j in range(jrange):
            Ib = imin[i*B:(i+1)*B, j*B:(j+1)*B]
            [label32[i, j], Noise_32[i, j]] = PCANoiseLevelEstimator(Ib, 5)
    MEDNoise_32 = medfilt(Noise_32, [5, 5])
    Noise_32[label32 == 1] = MEDNoise_32[label32 == 1]
    [u, re] = KMeans(Noise_32.flatten(order='F'), 2)
    irange = int(M/64)
    jrange = int(N/64)
    Noise_mix = cp.zeros((irange*2, jrange*2))
    initialdetected = cp.zeros((irange*2, jrange*2))
    for i in range(irange):
        for j in range(jrange):
            Noise_mix[2*i:2*(i+1), 2*j:2*(j+1)] = Noise_64[i, j]
            initialdetected[2*i:2*(i+1), 2*j:2*(j+1)] = result4[i, j]
    Noise_mix = 0.8*Noise_mix+0.2*Noise_32[:2*(i+1), :2*(j+1)]
    Noise_mix2 = Noise_mix.copy()
    DL = initialdetected[1:-1, :-2] - initialdetected[1:-1, 1:-1]
    DR = initialdetected[1:-1, 1:-1] - initialdetected[1:-1, 2:]
    DU = initialdetected[:-2, 1:-1] - initialdetected[1:-1, 1:-1]
    DD = initialdetected[1:-1, 1:-1] - initialdetected[2:, 1:-1]
    Edge = cp.zeros(cp.shape(initialdetected))
    Edge[1:-1, 1:-1] = cp.abs(DL)+cp.abs(DR)+cp.abs(DU)+cp.abs(DD)
    g = [Edge > 0]
    Noise_mix2[tuple(g)] = Noise_32[tuple(g)]
    [u, re] = KMeans(Noise_mix2.flatten(order='F'), 2)
    result4 = cp.reshape(re[:, 1], cp.shape(Noise_mix2), order='F')
    labels = cv2.connectedComponentsWithStats(cp.uint8(result4-1))
    bwpp = labels[1]
    area = labels[2][:, 4]
    for num in range(1, len(area)):
        if (area[num] < 4):
            result4[bwpp == num] = 1
    bwpp = cv2.connectedComponents(cp.uint8(result4-1))[1]
    return [Noise_mix2, bwpp.astype("uint8")]
