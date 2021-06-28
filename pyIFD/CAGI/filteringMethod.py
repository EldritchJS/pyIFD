import numpy as np

blocksize = 6

def filteringMethod (smap, ThressSmall, ThressBigV, ThressImg):

    blocks=np.size(smap,0)
    step=int(np.sqrt(blocks))
 
    ThressBig=np.ndarray.flatten(ThressBigV)
    for x in range(blocksize):
        if ((ThressBig[x]<ThressImg) and (ThressImg<10)):
            ThressBig[x]=ThressImg
        elif ((ThressBig[x]>ThressImg) and (ThressImg<5)):
            ThressBig[x]=5
        for y in range(blocksize):
            if (ThressSmall[x,y]<ThressBig[x]):
                if (ThressBig[x]<5):
                    ThressSmall[x,y]=ThressBig[x]+1
                else:
                    ThressSmall[x,y]=ThressBig[x]
    Thresses=ThressSmall;
    increment = int(step/blocksize)
    for a in range(1,blocksize+1):
        Start=int(a*(blocks/blocksize)-(blocks/blocksize))
        End=int(a*(blocks/blocksize))-1
        
        for x in range(Start,End,step): 
            for y in range(increment):
                z=x+y
                if (a<4):
                    if (smap[z,1]< Thresses[0,(a*2)-2]):
                        smap[z,0]=59
                    if smap[z+increment,1]<Thresses[0,a*2-1]:
                        smap[z+increment,0]=59
                    if smap[z+2*(increment),1]<Thresses[1,(a*2)-2]:
                        smap[z+2*(increment),0]=59
                    if smap[z+3*(increment),1]<Thresses[1,a*2-1]:
                        smap[z+3*(increment),0]=59
                    if smap[z+4*(increment),1]<Thresses[2,(a*2)-2]:
                        smap[z+4*(increment),0]=59
                    if smap[z+5*(increment),1]<Thresses[2,a*2-1]:
                        smap[z+5*(increment),0]=59
                else:
                    if smap[z,1]< Thresses[3,((a-3)*2)-2]:
                        smap[z,0]=59
                    if smap[z+increment,1]<Thresses[3,(a-3)*2-1]:
                        smap[z+increment,0]=59
                    if smap[z+2*(increment),1]<Thresses[4,((a-3)*2)-2]:
                        smap[z+2*(increment),0]=59
                    if smap[z+3*(increment),1]<Thresses[4,(a-3)*2-1]:
                        smap[z+3*(increment),0]=59
                    if smap[z+4*(increment),1]<Thresses[5,((a-3)*2)-2]:
                        smap[z+4*(increment),0]=59;
                    if smap[z+5*(increment),1]<Thresses[5,(a-3)*2-1]:
                        smap[z+5*(increment),0]=59
    return smap                        


