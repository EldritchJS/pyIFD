function [V] = localNoiVarEstimate_hdd(noi,ft,fz,br)
    # Markos Zampoglou: this is a variant of the original
    # localNoiVarEstimate.m, aimed to be more memory-efficient. The
    # original has been renamed to localNoiVarEstimate_ram
    #
    # localNoiVarEstimate: local noise variance estimation using kurtosis
    #
    # [estVar] = localNoiVarEstimate(noisyIm,filter_type,filter_size,block_size)
    #
    # input arguments:
    #	noisyIm: input noisy image
    #	filter_type: the type of band-pass filter used
    #        supported types, "dct", "haar", "rand"
    #   filter_size: the size of the support of the filter
    #   block_rad: the size of the local blocks
    # output arguments:
    #	estVar: estimated local noise variance
    #
    # reference:
    #   X.Pan, X.Zhang and S.Lyu, Exposing Image Splicing with
    #   Inconsistent Local Noise Variances, IEEE International
    #   Conference on Computational Photography, Seattle, WA, 2012
    #
    # disclaimer:
    #	Please refer to the ReadMe.txt
    #
    # Xunyu Pan, Xing Zhang and Siwei Lyu -- 07/26/2012
    
    if ft == 'dct':
        fltrs = dct2mtx(fz,'snake')
    elif ft == 'haar':
        fltrs = haar2mtx(fz)
    elif ft == 'rand':
        fltrs = rnd2mtx(fz)
    else:
        return 0

    # decompose into channels
    ch = np.zeros([size(noi),fz*fz-1],'single');
    for k = 2:(fz*fz)
        ch(:,:,k-1) = conv2(noi,fltrs(:,:,k),'same');
    end
    
    # collect raw moments
    blksz = (2*br+1)*(2*br+1)
    mu1 = block_avg(ch,br,'mi')
    mu2 = block_avg(ch**2,br,'mi');
    mu3 = block_avg(ch**3,br,'mi');
    mu4 = block_avg(ch**4,br,'mi');
    
    # variance & sqrt of kurtosis
    
    Factor34=mu4 - 4*mu1.*mu3;
    
    noiV = mu2 - mu1.**2
    noiK = (Factor34 + 6*mu1**2*mu2 - 3*mu1**4)/(noiV**2)-3
    noiK = np.max(0,noiK)
    
    
    
    a = np.mean(np.sqrt(noiK),3)
    b = np.mean(1/noiV,3)
    c = np.mean(1/noiV.**2,3)
    d = np.mean(np.sqrt(noiK)/noiV,3)
    e = np.mean(noiV,3);
    
    sqrtK = (a*c - b*d)/(c-b*b)
    
    V = ((1 - a/sqrtK)/b)
    idx = sqrtK<np.median(sqrtK)
    V[idx] = 1/b[idx]
    idx = V<0
    V[idx] = 1/b[idx]
    
    return V
