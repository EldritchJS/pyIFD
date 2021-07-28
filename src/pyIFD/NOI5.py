"""
This module provides the NOI5 algorithm
"""

import numpy as np
from numpy.linalg import eigh
import cv2
from scipy.ndimage import median_filter as medfilt

def KMeans(data,N):
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
    u=np.zeros((N,1));
    Sdata = np.sort(data);
    u[0] = np.mean(Sdata[-round(m/4)-1:])
    u[1] = np.mean(Sdata[:round(m/4)])
    umax = np.median(Sdata[-round(m/10)-1:])
    data[data>umax]= umax
    for iter in range(200):
        pre_u=u.copy()     #center of the last iter
        tmp=np.zeros((N,m))
        for i in range(N):
            tmp[i,:]=data-u[i]
        tmp = np.abs(tmp)
        junk=np.min(tmp,axis=0)
        index=np.argmin(tmp,axis=0)
        quan=np.zeros((m,N))
        for i in range(m):          
            quan[i,index[i]]=junk[i]
        for i in range(N): 
            if (np.sum(quan[:,i])>0.01):
                u[i]= np.sum(quan[:,i]*data)/np.sum(quan[:,i]);
        
        if (np.linalg.norm(pre_u-u)<0.02): 
            break;
    
    re=np.zeros((m,2))
    for i in range(m):
        tmp=np.zeros((N,1))
        for j in range(N):
            tmp[j]=np.linalg.norm(data[i]-u[j])
        
        junk=np.min(tmp,axis=0)
        index=np.argmin(tmp,axis=0)
        re[i,0]=data[i]
        re[i,1]=index+1
    # the tampered area is less than half of the whole image
    label = re[:,1]
    if list(label).count(1)<int(m/2):
        re[:,1]=3-label
    
    return [u,re]

def PCANoiseLevelEstimator( image, Bsize ):
    """
    Summary please.

    Args:
        image:
        Bsize:

    Returns:
        label:
        variance: 
    """
    UpperBoundLevel             = 0.0005
    UpperBoundFactor            = 3.1
    M1                          = Bsize
    M2                          = Bsize
    M                           = M1 * M2
    EigenValueCount             = 7
    EigenValueDiffThreshold     = 49.0
    LevelStep                   = 0.05
    MinLevel                    = 0.06
    MaxClippedPixelCount        = round(np.nextafter(0.1*M,0.1*M+1))
    
    #==========================================================================
    def Clamp(x, a, b):
        """
        Limit input value to a range.

        Args:
            x: value to clamp
            a: minimum value
            b: maximum value

        Returns:
            y: clamped value
        """        
        y=x
        if x < a:
            y = a
        if x > b:
            y = b
        return y

    #==========================================================================
    def ComputeBlockInfo(image):
        """
        Summary please.

        Args:
            image:

        Returns:
            block_info: 
        """        
        block_info = np.zeros((np.shape(image)[0]*np.shape(image)[1],3))
        block_count = 0

        for y in range(1,np.shape(image)[0] - M2+1):
            for x in range(1,np.shape(image)[1] - M1+1):  
                sum1 = 0.0
                sum2 = 0.0
                clipped_pixel_count = 0;

                for by in range(y-1,y + M2 - 1):
                    for bx in range(x-1,x + M1 - 1):                        
                        val = image[by,bx]
                        sum1 += val
                        sum2 += val**2

                        if val == 0 or val == 255:
                            clipped_pixel_count += 1
                if clipped_pixel_count <= MaxClippedPixelCount:                   
                    block_info[block_count,0] = (sum2 - sum1*sum1/M) / M
                    block_info[block_count,1] = x
                    block_info[block_count,2] = y
                    block_count += 1 
        block_info=np.delete(block_info,slice(block_count,np.shape(image)[0]*np.shape(image)[1]),0)
        return block_info
    #==========================================================================
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
        loop_iters=len(np.arange(1,MinLevel,-0.05))
        sum1 = np.zeros((M,1,loop_iters))
        sum2 =  np.zeros((M,M,loop_iters))
        subset_size = np.zeros((loop_iters,1))
        subset_count = 0
        max_index=np.shape(block_info)[0]-1
        for p  in np.arange(1,MinLevel,-LevelStep):
            q = 0
            if p - LevelStep > MinLevel:
                q = p - LevelStep

            beg_index = Clamp( round(q*max_index+LevelStep/2) + 1, 1, max_index+1 )
            end_index = Clamp( round(p*max_index+LevelStep/2) + 1, 1, max_index+1 )
            curr_sum1 = np.zeros((M, 1))
            curr_sum2 = np.zeros((M,M))
            for k in range (int(beg_index)-1,int(end_index)-1):
                curr_x = int(block_info[k,1])
                curr_y = int(block_info[k,2])
                block = np.reshape( image[curr_y-1 : curr_y+M2-1, curr_x-1 : curr_x+M1-1], (M, 1),order='F' ).astype("double")
                curr_sum1 += block
                curr_sum2 +=  block * block.T
            subset_count += 1
            sum1[:,:,subset_count-1] = curr_sum1.copy()
            sum2[:,:,subset_count-1] = curr_sum2.copy()
            subset_size[subset_count-1] = end_index - beg_index
        for i in range(len(subset_size)-1,0,-1):
            sum1[:,:,i-1] += sum1[:,:,i]
            sum2[:,:,i-1] += sum2[:,:,i]
            subset_size[i-1] += subset_size[i]
        return [sum1,sum2,subset_size]
    #==========================================================================
    def ComputeUpperBound(block_info):
        """
        Summary please.

        Args:
            block_info:

        Returns:
            upper_bound: 
        """        
        max_index = np.shape(block_info)[0] - 1
        zero_idx=np.where(block_info[:,0]== 0)[0]
        if zero_idx.size==0:
            nozeroindex=round(UpperBoundLevel*max_index)
        else:
            nozeroindex = min(np.max(np.where(block_info[:,0]== 0)[0])+1,np.shape(block_info)[0]-1)
        index = Clamp(round(UpperBoundLevel*max_index) + 1, nozeroindex, np.shape(block_info)[0]-1)
        upper_bound = UpperBoundFactor * block_info[index,0]
        return upper_bound
    #==========================================================================
    def ApplyPCA( sum1, sum2, subset_size ):        
        """
        Summary please.

        Args:
            sum1:
            sum2:
            subset_size:

        Returns:
            eigh: 
        """                
        meanval = sum1 / subset_size
        cov_matrix = sum2 / subset_size - meanval * np.transpose(meanval)
        return eigh(cov_matrix)[0]
    #==========================================================================
    def GetNextEstimate( sum1, sum2, subset_size, prev_estimate, upper_bound ):
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
        variance = 0;       
        for i in range(len(subset_size)):
            eigen_value = ApplyPCA( sum1[:,:,i], sum2[:,:,i], subset_size[i])
            variance=eigen_value[0]
            if variance < 0.00001: #1e-5: 
                break;
            diff            = eigen_value[EigenValueCount-1] - eigen_value[0]
            diff_threshold  = EigenValueDiffThreshold * prev_estimate / subset_size[i]**0.5

            if( diff < diff_threshold and variance < upper_bound ):
                break;
        return variance

    #==========================================================================
   
    label = 0
    block_info = ComputeBlockInfo( image )
    if np.min(np.shape(block_info))==0:
        label = 1
        variance = np.var(image)
    else:
        idx=np.lexsort((block_info[:,2],block_info[:,0]))
        block_info = np.asarray([block_info[i,:] for i in idx])
        [sum1, sum2, subset_size] = ComputeStatistics( image, block_info )
        if subset_size[-1] == 0:
            label = 1
            variance = np.var(image)
        else:
            upper_bound = ComputeUpperBound( block_info )
            prev_variance = 0
            variance = upper_bound
            for iter in range(10):
                if( np.abs(prev_variance - variance) < 0.00001): 
                    break
                prev_variance = variance
                variance = GetNextEstimate( sum1, sum2, subset_size, variance, upper_bound )
            if variance < 0: 
                label = 1
                variance = np.var(image)
    variance = np.sqrt(variance)
    return [label, variance]

def PCANoise(impath):
    """
    Main driver for NOI5 algorithm.
    
    Args:
        impath: input image
    
    Returns:
        OutputMap: Output image

    Todos:
        * Fix the returns
    """
    B = 64
    I = cv2.cvtColor(cv2.imread(impath), cv2.COLOR_BGR2GRAY).astype("double")
    [M,N] = np.shape(I)
    I = I[:int(np.floor(M/B)*B),:int(np.floor(N/B)*B)]
    [M, N] = np.shape(I)
    im = I.copy()
    irange = int(np.floor(M/B))
    jrange = int(np.floor(N/B))
    Ib=np.zeros((irange,jrange))
    label64=np.zeros((irange,jrange))
    Noise_64=np.zeros((irange,jrange))

    for i in range(irange):
        for j in range(jrange):
            Ib = I[i*B:(i+1)*B,j*B:(j+1)*B]
            Noise_64[i,j] =  PCANoiseLevelEstimator(Ib,5)[1]
    [u,re]  = KMeans(Noise_64.flatten(order='F'),2)
    result4 = np.reshape(re[:,1],np.shape(Noise_64),order='F')

    
    B = 32
    irange = int(np.floor(M/B))
    jrange = int(np.floor(N/B))
    label32=np.zeros((irange,jrange))
    Noise_32=np.zeros((irange,jrange))
    for i  in range(irange):
        for j in range(jrange):
            Ib = I[i*B:(i+1)*B,j*B:(j+1)*B]
            [label32[i,j], Noise_32[i,j]] =  PCANoiseLevelEstimator(Ib,5)
    MEDNoise_32= medfilt(Noise_32,[5, 5])
    Noise_32[label32==1]= MEDNoise_32[label32==1]
    [u, re]=KMeans(Noise_32.flatten(order='F'),2)
    result2=np.reshape(re[:,1],np.shape(Noise_32),order='F')
    irange = int(M/64)
    jrange = int(N/64)
    Noise_mix=np.zeros((irange*2,jrange*2))
    initialdetected=np.zeros((irange*2,jrange*2))
    for i in range(irange):
        for j in range(jrange):
            Noise_mix[2*i:2*(i+1),2*j:2*(j+1)] = Noise_64[i,j]
            initialdetected[2*i:2*(i+1),2*j:2*(j+1)] = result4[i,j]
    Noise_mix = 0.8*Noise_mix+0.2*Noise_32[:2*(i+1),:2*(j+1)]
    Noise_mix2 = Noise_mix.copy()
    DL = initialdetected[1:-1,:-2] - initialdetected[1:-1,1:-1]
    DR = initialdetected[1:-1,1:-1] - initialdetected[1:-1,2:]
    DU = initialdetected[:-2,1:-1] - initialdetected[1:-1,1:-1]
    DD = initialdetected[1:-1,1:-1] - initialdetected[2:,1:-1]
    Edge = np.zeros(np.shape(initialdetected))
    Edge[1:-1,1:-1]= np.abs(DL)+np.abs(DR)+np.abs(DU)+np.abs(DD)
    g = [Edge>0]
    Noise_mix2[tuple(g)] = Noise_32[tuple(g)]
    [u,re]=KMeans(Noise_mix2.flatten(order='F'),2)
    result4=np.reshape(re[:,1],np.shape(Noise_mix2),order='F')
    labels=cv2.connectedComponentsWithStats(np.uint8(result4-1))
    bwpp=labels[1]
    area = labels[2][:,4]
    for num in range(1,len(area)):
        if (area[num] < 4):
            result4[bwpp==num]=1
    bwpp = cv2.connectedComponents(np.uint8(result4-1))[1]
    return [Noise_mix2,bwpp.astype("uint8")]
