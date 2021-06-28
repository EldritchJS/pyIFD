def dct2mtx(n,ord): 
# DCT2MTX: generating matrices corresponding to 2D-DCT transform.
#          
#
# [mtx] = dct2mtx(N)
# 
# input arguments:
#	N: size of 2D-DCT basis (N x N)
#  ord: order of the obtained DCT basis
#		'grid': as grid order (default)
#     'snake': as snake order
# output arguments:
#	mtx: 3D matrices of dimension (NxNxN^2)
#       mtx(:,:,k) is the kth 2D DCT basis of support size N x N
#
# Xunyu Pan, Xing Zhang, Siwei Lyu -- 07/26/2012             

    (cc,rr) = np.meshgrid(0:n-1)

    c = np.sqrt(2 / n) * np.cos(np.pi * (2*cc + 1) * rr / (2 * n))
    c[0,:] = c[0,:] / np.sqrt(2)

switch ord(1:2)
    if ord[0:1]=='gr':
		ord = np.reshape(0:n**2,n,n)
    elif ord[0:1]=='sn': # not exactly snake code,but close
		temp = cc+rr
        idx = np.argsort(temp) # TODO: Check this
		ord = reshape(idx,n,n);

    mtx = np.zeros((n,n,n*n))
    for i in range(0,n):
        for j in range(0,n):
		    mtx[:,:,ord[i,j]] = np.transpose(c[i,:])*c[j,:]

return mtx
