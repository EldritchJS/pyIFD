import numpy as np

def hist_adjust(arr,bins):
    [A,B]=np.histogram(arr,bins)
    for i in range(1,bins):
        count=np.count_nonzero(arr==B[i])
        A[i]-=count
        A[i-1]+=count
    return [A,B]

def inblockpatterns(image, bins, p, q, blk_idx, blk_idy):
    Zmat=np.zeros((int(np.floor(blk_idx*blk_idy)),2))
    a=-1
    BlockScoreAll = np.zeros((blk_idx,blk_idy))
    for i in range(blk_idx):
        Ax=(i*8)+p-1
        Ex=Ax+4
        for j in range(blk_idy):
            Ay=(j*8)+q-1
            A=image[Ax,Ay]
            B=image[Ax, Ay+1]
            C=image[Ax+1, Ay]
            D=image[Ax+1, Ay+1]
            
            Ey=Ay+4
            E=image[Ex, Ey]
            F=image[Ex, Ey+1]
            G=image[Ex+1, Ey]
            H=image[Ex+1, Ey+1]
            
            a+=1
            
            Zmat[a,0]=abs(A-B-C+D)
            Zmat[a,1]=abs(E-F-G+H)
            
            BlockScoreAll[i,j] = Zmat[a,1] - Zmat[a,0]
            if (BlockScoreAll[i,j]<=0):
                BlockScoreAll[i,j]=0
    norm=a
    #Currently mismatched hist fcn
    Hz=hist_adjust(Zmat[:,0],bins)[0]
    Hzn=Hz/(norm+1)
    Hz2=hist_adjust(Zmat[:,1],bins)[0]
    Hz2n=Hz2/(norm+1)
    y2=int(Hzn.size)
    K=0
    for i in range(y2):
        K_temp=Hzn[i]-Hz2n[i]
        K+=abs(K_temp)
        
        
    A=sum(Hzn[0:2]);

    E=sum(Hz2n[0:2]);

    if A>E:
        Correct=True;
    else:
        Correct=False;

    return [K,Correct, BlockScoreAll]


#def hist_adjust2(arr,bins):
#    edges=(max(arr)-min(arr))/bins*range(bins+1)
#    for i in range(1,bins+1):
#        edges[i]+=np.spacing(edges[i])
#    return np.histogram(arr,edges)


#def histc(X, bins):
#    map_to_bins = np.digitize(X,bins)
#    r = np.zeros(bins.shape)
#    for i in map_to_bins:
#        r[i-1] += 1
#    return [r, map_to_bins]

#def hist_adjust3(arr,bins):
#    edges=np.linspace(min(arr),max(arr),bins+1)
#    edges+=np.spacing(edges)
#    edges[0]=-np.Inf
#    edges[-1]=np.Inf
#    return (histc(Zmat[:,0],edges)[0]).astype(int)
