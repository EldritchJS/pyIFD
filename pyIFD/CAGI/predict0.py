import numpy as np

def predict0(Kscores):

    Kpredict=np.zeros((9,9))
    Kpredict[0:8,0:8]=Kscores[:,:,1]
    for i in range(8):
        Kpredict[8,i]=sum(Kpredict[:,i])
        Kpredict[i,8]=sum(Kpredict[i,:])
 
    Kpre=np.zeros((8,8))
    for i in range(8):
        for j in range(8):
            Kpre[i,j]=(Kpredict[i,8]+Kpredict[8,j])/16
    
    return [Kpredict, Kpre]