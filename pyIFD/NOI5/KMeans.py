import numpy as np

# Finished KMeans review
def KMeans(data,N):
    m = data.size
    u=np.zeros((N,1));
    Sdata = np.sort(data);
    u[0] = np.mean(Sdata[-round(m/4)-1:])
    u[1] = np.mean(Sdata[:round(m/4)])
    umax = np.median(Sdata[-round(m/10)-1:])
    data[data>umax]= umax
    for iter in range(200):
        pre_u=u.copy()     #center of the last iter
        tmp=np.zeros((N,m))
        for i in range(N):
            tmp[i,:]=data-u[i]
        tmp = np.abs(tmp)
        junk=np.min(tmp,axis=0)
        index=np.argmin(tmp,axis=0)
        quan=np.zeros((m,N))
        for i in range(m):          
            quan[i,index[i]]=junk[i]
        for i in range(N): 
            if (np.sum(quan[:,i])>0.01):
                u[i]= np.sum(quan[:,i]*data)/np.sum(quan[:,i]);
        
        if (np.linalg.norm(pre_u-u)<0.02): 
            break;
    
    re=np.zeros((m,2))
    for i in range(m):
        tmp=np.zeros((N,1))
        for j in range(N):
            tmp[j]=np.linalg.norm(data[i]-u[j])
        
        junk=np.min(tmp,axis=0)
        index=np.argmin(tmp,axis=0)
        re[i,0]=data[i]
        re[i,1]=index+1
    # the tampered area is less than half of the whole image
    label = re[:,1]
    if list(label).count(1)<int(m/2):
        re[:,1]=3-label
    
    return [u,re]


