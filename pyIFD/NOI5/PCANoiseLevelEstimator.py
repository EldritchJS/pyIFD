import numpy as np
from numpy.linalg import eigvals
#Finished review of PCANoiseLevelEstimator
def PCANoiseLevelEstimator( image, Bsize ):
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
    #Finished Clamp review
    def Clamp(x, a, b):
        y=x
        if x < a:
            y = a
        if x > b:
            y = b
        return y

    #==========================================================================
    #Finished ComputeBlockInfo review
    def ComputeBlockInfo( image ):
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
    #Finished review of Compute Statistics
    def ComputeStatistics( image, block_info ):
        loop_iters=len(np.arange(1,MinLevel,-0.05))
        sum1 = np.zeros((M,1,loop_iters))
        sum2 =  np.zeros((M,M,loop_iters))
        subset_size = np.zeros((loop_iters,1))
        subset_count = 0

        for p  in np.arange(1,MinLevel,-LevelStep):
            q = 0
            if p - LevelStep > MinLevel:
                q = p - LevelStep

            max_index = np.shape(block_info)[0] - 1
            beg_index = Clamp( round(np.nextafter(q*max_index,q*max_index+1)) + 1, 1, np.shape(block_info)[0] )
            end_index = Clamp( round(np.nextafter(p*max_index,p*max_index+1)) + 1, 1, np.shape(block_info)[0] )
            curr_sum1 = np.zeros((M, 1))
            curr_sum2 = np.zeros((M,M))
            for k in range (int(beg_index)-1,int(end_index)-1):
                curr_x = int(block_info[k,1])
                curr_y = int(block_info[k,2])
                block = np.reshape( image[curr_y-1 : curr_y+M2-1, curr_x-1 : curr_x+M1-1], (M, 1),order='F' ).astype("double")
                curr_sum1 += block
                curr_sum2 +=  block * np.transpose(block)
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
    #Finished ComputeUpperBound review
    def ComputeUpperBound( block_info ):
        max_index = np.shape(block_info)[0] - 1
        zero_idx=np.where(block_info[:,0]== 0)[0]
        if zero_idx.size==0:
            nozeroindex=round(UpperBoundLevel*max_index)
        else:
            nozeroindex = np.min(np.max(np.where(block_info[:,0]== 0)[0])+1,np.shape(block_info)[0])
        index = Clamp(round(UpperBoundLevel*max_index) + 1, nozeroindex, np.shape(block_info)[0])
        upper_bound = UpperBoundFactor * block_info[index,0]
        return upper_bound
    #==========================================================================
    #Finished ApplyPCA review
    def ApplyPCA( sum1, sum2, subset_size ): 
            meanval = sum1 / subset_size
            cov_matrix = sum2 / subset_size - meanval * np.transpose(meanval)
            eigen_value = np.sort( eigvals(cov_matrix) )
            return eigen_value
    #==========================================================================
    #Finished GetNextEstimate review
    def GetNextEstimate( sum1, sum2, subset_size, prev_estimate, upper_bound ):
        variance = 0;       
        for i in range(len(subset_size)):
            eigen_value = ApplyPCA( sum1[:,:,i], sum2[:,:,i], subset_size[i])
            variance = eigen_value[0]
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
    if np.max(np.shape(block_info))==0:
        label = 1
        variance = np.var(image)
    else:
        idx=np.lexsort((block_info[:,2],block_info[:,0]))
        block_info = np.asarray([block_info[i,:] for i in idx])
        [sum1, sum2, subset_size] = ComputeStatistics( image, block_info );
        if subset_size[-1] == 0:
            label = 1
            variance = np.var(image)
        else:
            upper_bound = ComputeUpperBound( block_info )
            prev_variance = 0
            variance = upper_bound
    
            for iter in range(10):
                if( np.abs(prev_variance - variance) < 0.00001): #1e-5 ):
                    break
                prev_variance = variance
                variance = GetNextEstimate( sum1, sum2, subset_size, variance, upper_bound )
            if variance < 0:                
                label = 1
                variance = np.var(image)
    variance = np.sqrt(variance);
    return [label, variance]