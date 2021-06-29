import numpy as np
#from bdct import bdct
def ExtractYDCT( im ):
    im=np.double(im)
    
    Y=0.299*im[:,:,0]+0.587*im[:,:,1]+0.114*im[:,:,2]
    Y=Y[:int(np.floor(np.shape(Y)[0]/8)*8),:int(np.floor(np.shape(Y)[1]/8)*8)]
    Y-=128
    
    #T = dctmtx(8);
    #dct = @(block_struct) T * block_struct.data * T';
    #YDCT=round(blockproc(Y,[8 8],dct));
    #Use the command below instead of the one above, it's a tiny bit closer
    #to original JPEG DCT
    YDCT=np.round(bdct(Y,8)).astype("int")
    return YDCT


