import numpy as np

def SmapIng(ImgTiles, MaskTiles, WhiteMaskPoints):
    blocks=np.shape(ImgTiles)[2]
    smap=np.zeros((blocks,2))
    winMask=59
    MaskWhite = (MaskTiles>0).astype(int)
    MaskBlack = (MaskTiles<=0).astype(int)

    for a in range(blocks): #3600
        maxR=0
        for k in range(58): #58
            #TempW=0
            #TempB=0
            #for x in range(10): #10
            #    for y in range(10): #10
            #        if (MaskTiles[x,y,k]>0):
            #            TempW=TempW+ImgTiles[x,y,a]
            #        else: 
            #            TempB=TempB+ImgTiles[x,y,a]
            #TempW = np.sum([ImgTiles[x,y,a] for x in range(10) for y in range(10) if MaskTiles[x,y,k]>0])
            #TempB = np.sum([ImgTiles[x,y,a] for x in range(10) for y in range(10) if MaskTiles[x,y,k]<=0])
            TempW = np.sum(ImgTiles[:,:,a]*MaskWhite[:,:,k])
            TempB = np.sum(ImgTiles[:,:,a]*MaskBlack[:,:,k])
            whiteScore=TempW/WhiteMaskPoints[k]
            blackScore=TempB/(100-WhiteMaskPoints[k])
            ctR=np.abs(whiteScore-blackScore)
            w=((ctR*100)/255)
            if (w>maxR):
                maxR=w
                winMask=k+1

        smap[a,0]=winMask
        smap[a,1]=maxR

    return smap
