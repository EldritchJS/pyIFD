import numpy as np
import cv2
from scipy.signal import medfilt2d
from KMeans import KMeans
from PCANoiseLevelEstimator import PCANoiseLevelEstimator
from dethighlightHZ import dethighlightHZ

def NOI5(impath):
    B = 64
    I = cv2.cvtColor(cv2.imread("../demo.tif"), cv2.COLOR_BGR2GRAY).astype("double")
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
