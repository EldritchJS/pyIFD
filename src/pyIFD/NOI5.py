import numpy as np
from numpy.linalg import eigvals
import cv2
from scipy.signal import medfilt2d

# Finished KMeans review
def KMeans(data,N):
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

def dethighlightHZ(im,blocksize,detections):
    im = np.transpose(im)
    rval = 255
    bval = 0
    gval = 0
    
    if im.ndim==2:
        [rows,cols]=np.shape(im)
        colors=1
    else:
        [rows, cols, colors]= np.shape(im)
    rowblocks = int(np.floor(rows/blocksize))
    # calculate the number of blocks contained in the colnum
    colblocks = int(np.floor(cols/blocksize))
    # calculate the number of blocks contained in the rownum %cols/blocksize;
    if colors == 1:
        newim=np.zeros((rows,cols,3))
        newim[:,:,0]= im
        newim[:,:,1]= im
        newim[:,:,2]= im
        im = newim
    # pick red color layer for highlighting
    highlighted= im;

    for rowblock in range(1,rowblocks+1):
        for colblock in range(1,colblocks+1):
            if detections[rowblock-1,colblock-1] == 2:
            # label 2 in Kmeans denotes tampered area
                rowst= int((rowblock-1) * blocksize+1)
                rowfin= int(rowblock * blocksize)
                colst= int((colblock-1) * blocksize + 1)
                colfin= int(colblock * blocksize)
                # red
                highlighted[rowst-1:rowst+2,colst-1:colfin,0]= rval
                highlighted[rowfin-3:rowfin,colst-1:colfin,0]= rval
                highlighted[rowst-1:rowfin,colst-1:colst+2,0]= rval
                highlighted[rowst-1:rowfin,colfin-3:colfin,0]= rval
            
                # green
                highlighted[rowst-1:rowst+2,colst-1:colfin,1]= gval
                highlighted[rowfin-3:rowfin,colst-1:colfin,1]= gval
                highlighted[rowst-1:rowfin,colst-1:colst+2,1]= gval
                highlighted[rowst-1:rowfin,colfin-3:colfin,1]= gval
            
                # blue
                highlighted[rowst-1:rowst+2,colst-1:colfin,2]= bval
                highlighted[rowfin-3:rowfin,colst-1:colfin,2]= bval
                highlighted[rowst-1:rowfin,colst-1:colst+2,2]= bval
                highlighted[rowst-1:rowfin,colfin-3:colfin,2]= bval
            
                if rowst-1 > 0:
                    highlighted[rowst-4:rowst-1,colst-1:colfin,0]= rval
                    highlighted[rowst-4:rowst-1,colst-1:colfin,1]= gval
                    highlighted[rowst-4:rowst-1,colst-1:colfin,2]= bval
               
                    if colst-1 > 0:
                        highlighted[rowst-4:rowst-1,colst-4:colst-1,0]= rval
                        highlighted[rowst-4:rowst-1,colst-4:colst-1,1]= gval
                        highlighted[rowst-4:rowst-1,colst-4:colst-1,2]= bval 
                    if colfin+1 < cols:
                        highlighted[rowst-4:rowst-1,colfin:colfin+3,0]= rval
                        highlighted[rowst-4:rowst-1,colfin:colfin+3,1]= gval
                        highlighted[rowst-4:rowst-1,colfin:colfin+3,2]= bval
            
                if rowfin+1 < rows:
                    highlighted[rowfin:rowfin+3,colst-1:colfin,0]= rval
                    highlighted[rowfin:rowfin+3,colst-1:colfin,1]= gval
                    highlighted[rowfin:rowfin+3,colst-1:colfin,2]= bval
                
                    if colst-1 > 0:
                        highlighted[rowfin:rowfin+3,colst-4:colst-1,0]= rval
                        highlighted[rowfin:rowfin+3,colst-4:colst-1,1]= gval
                        highlighted[rowfin:rowfin+3,colst-4:colst-1,2]= bval
                    if colfin+1 < cols:
                        highlighted[rowfin:rowfin+3,colfin:colfin+3,0]= rval
                        highlighted[rowfin:rowfin+3,colfin:colfin+3,1]= gval
                        highlighted[rowfin:rowfin+3,colfin:colfin+3,2]= bval
           
                if colst-1 > 0:
                    highlighted[rowst-1:rowfin,colst-4:colst-1,0]= rval
                    highlighted[rowst-1:rowfin,colst-4:colst-1,1]= gval
                    highlighted[rowst-1:rowfin,colst-4:colst-1,2]= bval
            
                if colfin+1 < cols:
                    highlighted[rowst-1:rowfin,colfin:colfin+3,0]= rval
                    highlighted[rowst-1:rowfin,colfin:colfin+3,1]= gval
                    highlighted[rowst-1:rowfin,colfin:colfin+3,2]= bval
    highlighted= highlighted.astype(np.uint8)
    out=np.zeros((cols,rows,3))
    for i in range(3):
        out[:,:,i]=np.transpose(highlighted[:,:,i])
    return out

def PCANoise(impath):
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
            (label64[i,j], Noise_64[i,j]) =  PCANoiseLevelEstimator(Ib,5)
    [u,re]  = KMeans(Noise_64.flatten(order='F'),2)
    result4 = np.reshape(re[:,1],np.shape(Noise_64),order='F') # trace to determine size
    

    
    B = 32
    irange = int(np.floor(M/B))
    jrange = int(np.floor(N/B))
    label32=np.zeros((irange,jrange))
    Noise_32=np.zeros((irange,jrange))
    for i  in range(irange):
        for j in range(jrange):
            Ib = I[i*B:(i+1)*B,j*B:(j+1)*B]
            [label32[i,j], Noise_32[i,j]] =  PCANoiseLevelEstimator(Ib,5)
    MEDNoise_32= medfilt2d(Noise_32,[5, 5])
    Noise_32[label32==1]= MEDNoise_32[label32==1]
    [u, re]=KMeans(Noise_32.flatten(order='F'),2)
    result2=np.reshape(re[:,1],np.size(Noise_32),order='F') # trace to determine size
    irange = int(np.floor(M/64))
    jrange = int(np.floor(N/64))
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
    bwpp = cv2.connectedComponents(np.uint8(result4-1)) 
    highlighted=dethighlightHZ(I,B,np.transpose(result4)).astype("uint8")

    return [Noise_mix2,highlighted]
