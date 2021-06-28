import numpy as np

def RescaleToImageResult(E,sgrid,kx,ky,pixels):
    result = np.zeros((kx*sgrid*8,ky*sgrid*8))
    for x in range(kx):
        for y in range(ky):
            a=x*sgrid*8
            b=y*sgrid*8
            result[a:a+sgrid*8,b:b+sgrid*8]  = E[x,y]
    [xim, yim]=pixels.shape
    [xres, yres]=result.shape
    Result=np.zeros((xim,yim))
    Result[:xres,:yres]=result

    for  k in range(xres,xim):
        for y in range(yres):
            Result[k, y]=result[xres-1,y]

    for k in range(xim):
        for y in range(yres,yim):
            Result[k, y]=Result[k,yres-1] # TODO Review yres here

    return Result