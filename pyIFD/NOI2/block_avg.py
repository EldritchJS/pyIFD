import numpy as np
def block_avg(X,d,pad='zero'): 
# BLOCK_SUM: Compute the avg of elements for all overlapping dxd windows
#            in data X, where d = 2*rad+1.
#
# [bksum] = block_avg(X,rad)
# 
# input arguments:
    #X: an [nx,ny,ns] array as a stack of ns images of size [nx,ny]
    #rad: radius of the sliding window, i.e., window size = (2*rad+1)*(2*rad+1)
#  pad: padding patterns:
            #'zero': padding with zeros (default)
            #'mirror': padding with mirrored boundary area
#
# output arguments:
#    bksum:sum of elements for all overlapping dxd windows
#
# Xunyu Pan, Xing Zhang and Siwei Lyu -- 07/26/2012             

    [nx,ny,ns] = np.shape(X)
    if d < 0 or d != np.floor(d) or d >= min(nx,ny):
    #error('window size needs to be a positive integer');
        return
    wd = 2*d+1 # size of the sliding window

    Y = np.zeros((nx+wd,ny+wd,ns),'single')
    Y[d+1:nx+d+1,d+1:ny+d+1,:] = X 

    # padding boundary
    if pad[0:2] != 'ze':
        if pad[0:2] == 'mi':
    # padding by mirroring
        # mirroring top
            Y[1:d+1,:,:] = np.flip(Y[d+2:wd+1,:,:],axis=0)
            # mirroring bottom
            Y[nx+d+1:,:,:] = np.flip(Y[nx:nx+d,:,:],axis=0)
            # mirroring left
            Y[:,1:d+1,:] = np.flip(Y[:,d+2:wd+1,:],axis=1)
            # mirroring right
            Y[:,ny+d+1:,:] = np.flip(Y[:,ny:ny+d,:],axis=1)
        else:
            #error('unknown padding pattern');
            return
    
# forming integral image
    Y = np.cumsum(np.cumsum(Y,0),1)

    # computing block sums
    Y = Y[wd:,wd:,:]+Y[:-wd,:-wd,:] - Y[wd:,:-wd,:]-Y[:-wd,wd:,:]
    Y /=(wd*wd)
    return Y 