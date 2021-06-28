def haar2mtx(n): 
# DCT2MTX: generating matrices corresponding to 2D-Haar wavelet transform.
#          
#
# [mtx] = haar2mtx(N)
# 
# input arguments:
#	N: size of 2D-DCT basis (N x N)
#	   has to be power of two
# output arguments:
#	mtx: 3D matrices of dimension (NxNxN^2)
#       mtx(:,:,k) is the kth 2D haar basis of support size N x N
#
# Xunyu Pan, Xing Zhang, Siwei Lyu -- 07/26/2012             
#
# code borrowed from Jin Qi's HaarTransforamtionMatrix function

# check input parameter and make sure it's the power of 2
    Level=np.log2(n)
    if 2**Level<n:
        return
#     error('input parameter has to be the power of 2');

#Initialization
    c=[1]
    NC=1/np.sqrt(2) #normalization constant
    LP=[1 1]
    HP=[1 -1]

    # iteration from H=[1] 
    for i in range(0,Level):
        c = NC*[np.kron(c,LP);np.kron(np.eye(np.shape(c)),HP)]


    mtx = np.zeros((n,n,n*n))
    k = 0
    for i in range(0,n):
        for j in range(0,n):
	        mtx[:,:,k] = np.transpose(c[i,])*c[j,:]
	        k = k+1
    return mtx
