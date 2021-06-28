import numpy as np

blocksize = 6

def filtering(smap):
    blocks=np.shape(smap)[0]
    step=np.int(np.sqrt(blocks))
    smallAreas=np.zeros((blocksize,blocksize))
    increment=np.int(step/blocksize)
    for a in range(blocksize):
        Start=np.int((a+1)*(blocks/blocksize)-(blocks/blocksize)+1)
        End=np.int((a+1)*(blocks/blocksize))
        for x in range(Start,End,step):
            for y in range(increment):
                z=x+y-1
                if (a<3):
                    smallAreas[0,a*2]=smallAreas[0,a*2]+smap[z,1]
                    smallAreas[0,a*2+1]=smallAreas[0,a*2+1]+smap[z+increment,1]

                    smallAreas[1,(a*2)]=smallAreas[1,(a*2)]+smap[z+2*(increment),1]
                    smallAreas[1,a*2+1]=smallAreas[1,a*2+1]+smap[z+3*(increment),1]
                
                    smallAreas[2,a*2]=smallAreas[2,a*2]+smap[z+4*(increment),1]
                    smallAreas[2,a*2+1]=smallAreas[2,a*2+1]+smap[z+5*(increment),1]
                else: 
                    smallAreas[3,((a-3)*2)]=smallAreas[3,((a-3)*2)]+smap[z,1]
                    smallAreas[3,(a-3)*2+1]=smallAreas[3,(a-3)*2+1]+smap[z+increment,1]
                
                    smallAreas[4,((a-3)*2)]=smallAreas[4,((a-3)*2)]+smap[z+2*(increment),1]
                    smallAreas[4,(a-3)*2+1]=smallAreas[4,(a-3)*2+1]+smap[z+3*(increment),1]
                
                    smallAreas[5,((a-3)*2)]=smallAreas[5,((a-3)*2)]+smap[z+4*(increment),1]
                    smallAreas[5,(a-3)*2+1]=smallAreas[5,(a-3)*2+1]+smap[z+5*(increment),1]
    meansmallAreas=smallAreas/100
    meanbigAreas=np.zeros((1,blocksize))
    for x in range(blocksize):
        meanbigAreas[0,x]=np.mean(meansmallAreas[x,:])
    meanImg=np.mean(meanbigAreas)

    return [meansmallAreas, meanbigAreas, meanImg]
