import numpy as np
def EstimateJPEGQuality(imIn):
    if(len(imIn.quant_tables)==1):
        imIn.quant_tables[1]=imIn.quant_tables[0]
    YQuality=100-(np.sum(imIn.quant_tables[0])-imIn.quant_tables[0][0][0])/63
    CrCbQuality=100-(np.sum(imIn.quant_tables[1])-imIn.quant_tables[0][0][0])/63
    Diff=abs(YQuality-CrCbQuality)*0.98
    Quality=(YQuality+2*CrCbQuality)/3+Diff
    return Quality