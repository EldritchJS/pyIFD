import numpy as np

def dethighlightHZ(im,blocksize,detections):
    im = np.transpose(im)
    rval = 255
    bval = 0
    gval = 0
    
    if im.ndim==2:
        [rows,cols]=np.shape(im)
        colors=1
    else:
        [rows, cols, colors]= np.shape(im)
    rowblocks = int(np.floor(rows/blocksize))
    # calculate the number of blocks contained in the colnum
    colblocks = int(np.floor(cols/blocksize))
    # calculate the number of blocks contained in the rownum %cols/blocksize;
    if colors == 1:
        newim=np.zeros((rows,cols,3))
        newim[:,:,0]= im
        newim[:,:,1]= im
        newim[:,:,2]= im
        im = newim
    # pick red color layer for highlighting
    highlighted= im;

    for rowblock in range(1,rowblocks+1):
        for colblock in range(1,colblocks+1):
            if detections[rowblock-1,colblock-1] == 2:
            # label 2 in Kmeans denotes tampered area
                rowst= int((rowblock-1) * blocksize+1)
                rowfin= int(rowblock * blocksize)
                colst= int((colblock-1) * blocksize + 1)
                colfin= int(colblock * blocksize)
                # red
                highlighted[rowst-1:rowst+2,colst-1:colfin,0]= rval
                highlighted[rowfin-3:rowfin,colst-1:colfin,0]= rval
                highlighted[rowst-1:rowfin,colst-1:colst+2,0]= rval
                highlighted[rowst-1:rowfin,colfin-3:colfin,0]= rval
            
                # green
                highlighted[rowst-1:rowst+2,colst-1:colfin,1]= gval
                highlighted[rowfin-3:rowfin,colst-1:colfin,1]= gval
                highlighted[rowst-1:rowfin,colst-1:colst+2,1]= gval
                highlighted[rowst-1:rowfin,colfin-3:colfin,1]= gval
            
                # blue
                highlighted[rowst-1:rowst+2,colst-1:colfin,2]= bval
                highlighted[rowfin-3:rowfin,colst-1:colfin,2]= bval
                highlighted[rowst-1:rowfin,colst-1:colst+2,2]= bval
                highlighted[rowst-1:rowfin,colfin-3:colfin,2]= bval
            
                if rowst-1 > 0:
                    highlighted[rowst-4:rowst-1,colst-1:colfin,0]= rval
                    highlighted[rowst-4:rowst-1,colst-1:colfin,1]= gval
                    highlighted[rowst-4:rowst-1,colst-1:colfin,2]= bval
               
                    if colst-1 > 0:
                        highlighted[rowst-4:rowst-1,colst-4:colst-1,0]= rval
                        highlighted[rowst-4:rowst-1,colst-4:colst-1,1]= gval
                        highlighted[rowst-4:rowst-1,colst-4:colst-1,2]= bval 
                    if colfin+1 < cols:
                        highlighted[rowst-4:rowst-1,colfin:colfin+3,0]= rval
                        highlighted[rowst-4:rowst-1,colfin:colfin+3,1]= gval
                        highlighted[rowst-4:rowst-1,colfin:colfin+3,2]= bval
            
                if rowfin+1 < rows:
                    highlighted[rowfin:rowfin+3,colst-1:colfin,0]= rval
                    highlighted[rowfin:rowfin+3,colst-1:colfin,1]= gval
                    highlighted[rowfin:rowfin+3,colst-1:colfin,2]= bval
                
                    if colst-1 > 0:
                        highlighted[rowfin:rowfin+3,colst-4:colst-1,0]= rval
                        highlighted[rowfin:rowfin+3,colst-4:colst-1,1]= gval
                        highlighted[rowfin:rowfin+3,colst-4:colst-1,2]= bval
                    if colfin+1 < cols:
                        highlighted[rowfin:rowfin+3,colfin:colfin+3,0]= rval
                        highlighted[rowfin:rowfin+3,colfin:colfin+3,1]= gval
                        highlighted[rowfin:rowfin+3,colfin:colfin+3,2]= bval
           
                if colst-1 > 0:
                    highlighted[rowst-1:rowfin,colst-4:colst-1,0]= rval
                    highlighted[rowst-1:rowfin,colst-4:colst-1,1]= gval
                    highlighted[rowst-1:rowfin,colst-4:colst-1,2]= bval
            
                if colfin+1 < cols:
                    highlighted[rowst-1:rowfin,colfin:colfin+3,0]= rval
                    highlighted[rowst-1:rowfin,colfin:colfin+3,1]= gval
                    highlighted[rowst-1:rowfin,colfin:colfin+3,2]= bval
    highlighted= highlighted.astype(np.uint8)
    out=np.zeros((cols,rows,3))
    for i in range(3):
        out[:,:,i]=np.transpose(highlighted[:,:,i])
    return out

