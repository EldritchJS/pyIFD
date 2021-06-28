import numpy as np
def predict1(Kscores,Kpredict,Kpre):
    A = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            A[i,j]=Kscores[i,j,0] + Kscores[i+4,j+4, 0] -Kscores[i+4,j,0]-Kscores[i,j+4,0]

    r1=[1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4]
    c1=[1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4]

    #r1=[0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
    #c1=[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]


    PossiblePoints=np.zeros((len(r1),8));
    A_point = [0,0]
    E_point = [0,0]
    
    for i in range(len(r1)):
        r=r1[i]
        c=c1[i]
     
        if (A[r-1,c-1]>0): 
            if (Kpredict[r-1,c-1]==1):
                A_point[0]=r
                A_point[1]=c
                E_point[0]=r+4
                E_point[1]=c+4
            else:
                E_point[0]=r;
                E_point[1]=c;
                A_point[0]=r+4;
                A_point[1]=c+4;
        else:
            if (Kpredict[r-1,c+3]==1):
                A_point[0]=r
                A_point[1]=c+4;
                E_point[0]=r+4;
                E_point[1]=c;   
            else:
                E_point[0]=r;
                E_point[1]=c+4;
                A_point[0]=r+4;
                A_point[1]=c;  
        PossiblePoints[i,0]= A_point[0];
        PossiblePoints[i,1]= A_point[1];
        PossiblePoints[i,2]= E_point[0];
        PossiblePoints[i,3]= E_point[1];

        PossiblePoints[i,4]= Kscores[r-1,c-1,0]/2;
        PossiblePoints[i,5]=0;  
 
    for i in range(len(r1)):
        PossiblePoints[i,6]=Kpre[int(PossiblePoints[i,0])-1,int(PossiblePoints[i,1])-1] -Kpre[int(PossiblePoints[i,2])-1,int(PossiblePoints[i,3])-1]
        PossiblePoints[i,7]=(PossiblePoints[i,6]  + PossiblePoints[i,4])/2                    
   
    return PossiblePoints 