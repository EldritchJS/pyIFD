import numpy as np
import numpy.matlib
def rnd2mtx(n): 
# DCT2MTX: generating matrices corresponding to random orthnormal transform.
#
# [mtx] = rnd2mtx(N)
# 
# input arguments:
#	N: size of 2D random basis (N x N)
#
# output arguments:
#	mtx: 3D matrices of dimension (NxNxN^2)
#       mtx(:,:,k) is the kth 2D DCT basis of support size N x N
#
# Xunyu Pan, Xing Zhang, Siwei Lyu -- 07/26/2012             

    X = np.random.randn(n,n)
    X -= np.matlib.repmat(np.mean(X,0),n,1)
    X /=np.matlib.repmat(np.sqrt(np.sum(X**2,0)),n,1)

    mtx = np.zeros((n,n,n*n))
    k = 0 
    for i in range(n):
        for j in range(n):
            mtx[:,:,k] = np.outer(X[:,i],np.transpose(X[:,j]))
            k+=1
    return mtx