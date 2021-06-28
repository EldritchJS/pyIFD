import numpy as np

def PaintimgEdges(smap, MMasks, scale):

    if (scale==1):
        blocks=3600
        stepX=60
 
    edgeImg=np.zeros((600,600))
    edgeImg2=np.zeros((600,600))
    edgeImg3=np.zeros((600,600))
    countx=-1
    for a in range(stepX):
        for b in range(stepX):
            countx+=1
            i=-1
            for x in range(a*10,a*10+10):
                i+=1
                j=-1
                for y in range(b*10,b*10+10):
                    j+=1
                    edgeImg[x,y]=MMasks[i,j,int(smap[countx,0]-1)];
                    if (smap[countx,0]==59):
                        edgeImg3[x,y]=0
                    else:
                         edgeImg3[x,y]=1
                    edgeImg2[x,y]=smap[countx,1]
                    
    return [edgeImg2, edgeImg,edgeImg3]


