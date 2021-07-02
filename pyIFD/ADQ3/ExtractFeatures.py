import numpy as np
#from dequantize import dequantize
def ExtractFeatures(im,c1,c2,ncomp,digitBinsToKeep):
    coeffArray=im.coef_arrays[ncomp-1]
    qtable=im.quant_tables[im.comp_info[ncomp].quant_tbl_no-1]
    Y=dequantize(coeffArray,qtable)
    coeff=[1, 9, 2, 3, 10, 17, 25, 18, 11, 4, 5, 12, 19, 26, 33, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 57, 50, 43, 36, 29, 22, 15, 8, 16, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 24, 32, 39, 46, 53, 60, 61, 54, 47, 40, 48, 55, 62, 63, 56, 64]
    sizeCA=np.shape(coeffArray)
    digitHist=np.zeros((c2-c1+1,10))
    for index in range(c1,c2+1):
        coeffFreq=np.zeros((int(np.size(coeffArray)/64),))
        coe=coeff[index-1]
        k=1
        start=coe%8
        if start==0:
            start=8
        for l in range(start,sizeCA[1]+1,8):
            for i in range(int(np.ceil(coe/8)),sizeCA[0],8):
                coeffFreq[k-1]=Y[i-1,l-1]
                k+=1
        NumOfDigits=(np.floor(np.log10(abs(coeffFreq)+0.5))+1)
        tmp=[10**(i-1) for i in np.array(NumOfDigits)]
        FirstDigit=np.floor(np.divide(abs(coeffFreq),tmp)).astype("uint8")
        
        binHist=list(np.arange(0.5,9.5,1))
        binHist.insert(0,-float('Inf'))
        binHist.append(float('Inf'))
        digitHist[index-c1,:]=np.histogram(FirstDigit,binHist)[0]
    HistToKeep=digitHist[:,digitBinsToKeep]
    return np.ndarray.flatten(HistToKeep)

