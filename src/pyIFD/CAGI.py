"""
This module provides the CAGI algorithm

JPEG-grid-alignment-abnormality-based detector.

Algorithm attribution:
Iakovidou, Chryssanthi, Markos Zampoglou, Symeon Papadopoulos, and Yiannis Kompatsiaris. "Content-aware detection of JPEG grid inconsistencies for intuitive image forensics." Journal of Visual Communication and Image Representation 54 (2018): 155-170.

Based on code from:
Zampoglou, M., Papadopoulos, S., & Kompatsiaris, Y. (2017). Large-scale evaluation of splicing localization algorithms for web images. Multimedia Tools and Applications, 76(4), 4801–4834.
"""

from PIL import Image
from skimage.color import rgb2gray
import numpy as np
from scipy.ndimage import correlate
import cv2
import os


def im2double(im):
    """
    Converts image to type double.

    Args:
        im: Input image

    Returns:
        image as double: Converts type of im to double. Scales so elements lie from 0 to 1.
    """
    info = np.iinfo(im.dtype)
    return im.astype(np.double) / info.max


def ImageTiling(OImg):
    """
    Fill me in please.

    Args:
        OImg:

    Returns:
        tile:

    Todos:
        * Fill this in with proper summary
    """
    Img = np.array(Image.fromarray(OImg.astype(np.uint8)).resize(size=(600, 600), resample=Image.NEAREST))
    R1 = rgb2gray(Img)
    R = R1*255

    blocks = 3600
    stepX = 60
    stepY = 60
    ImgR = R.astype('int')

    countx = -1
    tile = np.zeros((10, 10, blocks))
    for a in range(stepX):
        for b in range(stepY):
            countx += 1
            i = -1
            for x in range((a*10), (a*10)+10):
                i += 1
                j = -1
                for y in range((b*10), (b*10)+10):
                    j += 1
                    tile[i, j, countx] = ImgR[x, y]
    return tile


def MainTrain(R10, blk_idx, blk_idy):
    """
    Fill me in please.

    Args:
        R10:
        blk_idx:
        blk_idy:

    Returns:
        MeanContent:
        MeanStrongEdge:

    Todos:
        * Fill this in with proper summary
    """
    [x, y, z] = R10.shape
    [PMasks, MMasks, MaskWhite] = getMasks()
    # ////////Image Tiling 3 Scales////////////////////////////
    # slight difference in tileF (~99% similarity)
    tileF = ImageTiling(R10)

    # ////////////Smaping/////////////////////////////////////
    smapF = SmapIng(tileF, PMasks, MaskWhite)
    # % % % %////////////Filtering///////////////////////////////////
    [ThresSmall, ThresBig, ThresImg] = filtering(smapF)
    smapF_filtrOld = filteringMethod(smapF, ThresSmall, ThresBig, ThresImg)
    # Through here so far
    # /////////////PaintEdges///////////////////////////////// This uses NN PIL using mean
    [e, edge, contours] = PaintimgEdges(smapF_filtrOld, MMasks, 1)
    Output = np.array(Image.fromarray(e.astype(np.double)).resize(size=(y, x), resample=Image.NEAREST))
    StrongEdge = np.array(Image.fromarray(contours.astype(np.uint8)).resize(size=(y, x), resample=Image.NEAREST))

    MeanContent = np.zeros((blk_idx, blk_idy))
    MeanStrongEdge = np.zeros((blk_idx, blk_idy))
    for i in range(blk_idx):
        for j in range(blk_idy):
            a = i*8
            b = j*8
            MeanContent[i, j] = np.mean(Output[a:a+8, b:b+8])
            MeanStrongEdge[i, j] = np.mean(StrongEdge[a:a+8, b:b+8])
    MeanStrongEdge[MeanStrongEdge > 0.5] = 1

    MeanStrongEdge[MeanStrongEdge <= 0.5] = 0
    return [MeanContent, MeanStrongEdge]


def PaintimgEdges(smap, MMasks, scale):
    """
    Fill me in please.

    Args:
        smap:
        MMasks:
        scale:

    Returns:
        edgeImg2:
        edgeImg:
        edgeImg3:

    Todos:
        * Fill this in with proper summary
    """
    if (scale == 1):
        stepX = 60

    edgeImg = np.zeros((600, 600))
    edgeImg2 = np.zeros((600, 600))
    edgeImg3 = np.zeros((600, 600))
    countx = -1
    for a in range(stepX):
        for b in range(stepX):
            countx += 1
            i = -1
            for x in range(a*10, a*10+10):
                i += 1
                j = -1
                for y in range(b*10, b*10+10):
                    j += 1
                    edgeImg[x, y] = MMasks[i, j, int(smap[countx, 0]-1)]
                    if (smap[countx, 0] == 59):
                        edgeImg3[x, y] = 0
                    else:
                        edgeImg3[x, y] = 1
                    edgeImg2[x, y] = smap[countx, 1]
    return [edgeImg2, edgeImg, edgeImg3]


def RescaleToImageResult(E, sgrid, kx, ky, pixels):
    """
    Fill me in please.

    Args:
        E:
        sgrid:
        kx:
        ky:
        pixels:

    Returns:
        Result:

    Todos:
        * Fill this in with proper summary
    """
    result = np.zeros((kx*sgrid*8, ky*sgrid*8))
    for x in range(kx):
        for y in range(ky):
            a = x*sgrid*8
            b = y*sgrid*8
            result[a:a+sgrid*8, b:b+sgrid*8] = E[x, y]
    [xim, yim] = pixels.shape
    [xres, yres] = result.shape
    Result = np.zeros((xim, yim))
    Result[:xres, :yres] = result

    for k in range(xres, xim):
        for y in range(yres):
            Result[k, y] = result[xres-1, y]

    for k in range(xim):
        for y in range(yres, yim):
            Result[k, y] = Result[k, yres-1]
    return Result


def SmapIng(ImgTiles, MaskTiles, WhiteMaskPoints):
    """
    Fill me in please.

    Args:
        ImgTiles:
        MaskTiles:
        WhiteMaskPoints:

    Returns:
        smap:

    Todos:
        * Fill this in with proper summary
    """
    blocks = np.shape(ImgTiles)[2]
    smap = np.zeros((blocks, 2))
    winMask = 59
    MaskWhite = (MaskTiles > 0).astype(int)
    MaskBlack = (MaskTiles <= 0).astype(int)

    for a in range(blocks):
        maxR = 0
        for k in range(58):
            TempW = np.sum(ImgTiles[:, :, a]*MaskWhite[:, :, k])
            TempB = np.sum(ImgTiles[:, :, a]*MaskBlack[:, :, k])
            whiteScore = TempW/WhiteMaskPoints[k]
            blackScore = TempB/(100-WhiteMaskPoints[k])
            ctR = np.abs(whiteScore-blackScore)
            w = ((ctR*100)/255)
            if (w > maxR):
                maxR = w
                winMask = k+1

        smap[a, 0] = winMask
        smap[a, 1] = maxR
    return smap


def mat2gray(A):
    """
    Converts matrix to have values from 0-1.

    Args:
        A: Input matrix.

    Returns:
        Gray matrix with values from 0-1.

    Todos:
        * Fill this in with proper summary
    """
    A -= A.min()
    if(A.max() == 0):
        return A
    return A/A.max()


def characterizeblocks(MeanContent2, MeanStrongEdge, V_im, blk_idx, blk_idy, MeanInSpace, diff_Mean_Best_scaled, dmbsi, sgrid, PossiblePoints, kx, ky):
    """
    Fill me in please.

    Args:
        MeanContent2:
        MeanStrongEdge:
        V_im:
        blk_idx:
        blk_idy:
        MeanInSpace:
        diff_Mean_Best_scaled:
        diff_Mean_Best_scaledInv (dmbsi):
        sgrid:
        PossiblePoints:
        kx:
        ky:

    Returns:
        E:
        EInv:

    Todos:
        * Fill this in with proper summary
    """
    uniform = np.zeros((int(np.floor(blk_idx/sgrid)), int(np.floor(blk_idy/sgrid))))

    for a in range(kx):
        for b in range(ky):
            for pp in range(16):
                if (MeanInSpace[a, b, pp] < (np.mean(MeanInSpace[:, :, pp])*0.2)):
                    uniform[a, b] += 1

    st = np.std(np.reshape(uniform, (uniform.size, 1), 'F'))
    H = np.ones((5, 5))*0.04

    im = correlate(uniform, H, mode='constant')
    meanv = np.mean(im)

    bg = 0
    for f in range(16):
        if ((PossiblePoints[f, 0] == 4) and (PossiblePoints[f, 1] == 4)):
            bg = f+1

    if bg == 16:
        bestgrid = mat2gray(correlate(MeanInSpace[:, :, 15], H, mode='constant'))
    elif bg == 0:
        bg1 = np.where(PossiblePoints[:, 4] == max(PossiblePoints[:, 4]))
        bg = np.max(bg1)+1
        bestgrid = mat2gray(correlate(MeanInSpace[:, :, bg-1], H, mode='constant'))
    else:
        bestgrid = mat2gray(correlate(MeanInSpace[:, :, bg], H, mode='constant'))
# //////////block based homogenous
    if ((np.mean(PossiblePoints[:, 4]) > 0.4) or (bg != 16)):
        homB = 0
    else:
        homB = 1

    if ((st/meanv) > 1.5):
        im[im < (meanv+(st))] = 0
        im[im >= (meanv+(st))] = homB
    else:
        im[im < (meanv+(st)/2)] = 0
        im[im >= (meanv+(st)/2)] = homB

# /////////no content////////////////////////

    contentsc = MeanContent2.copy()

    hom = np.zeros((kx, ky))
    for i in range(kx):
        for j in range(ky):
            if (contentsc[i, j] <= 4):  # very soft responses
                hom[i, j] = 1

    c = sgrid
    MeanStrongEdge2 = np.zeros((kx, ky))
    for i in range(kx):
        for j in range(ky):
            a = i*sgrid
            b = j*sgrid
            MeanStrongEdge2[i, j] = np.mean(MeanStrongEdge[a:a+c, b:b+c])
    cc = 8*sgrid
    V_im2 = np.zeros((kx, ky))
    for i in range(kx):
        for j in range(ky):
            a = i*8*sgrid
            b = j*8*sgrid
            V_im2[i, j] = np.mean(V_im[a:a+cc, b:b+cc])

    V_imOver = V_im2.copy()
    V_imUndr = V_im2.copy()
    V_imOver[V_imOver >= 245] = 300
    V_imOver[V_imOver != 300] = 0
    V_imUndr[V_imUndr < 15] = 300
    V_imUndr[V_imUndr != 300] = 0
    V_imOver = mat2gray(V_imOver)
    V_imUndr = mat2gray(V_imUndr)
    MeanStrongEdge2[MeanStrongEdge2 < 0.5] = 0
    MeanStrongEdge2[MeanStrongEdge2 >= 0.5] = 1
    # /////////////end overexposed/iunder and contours////////////////////

    touse = kx*ky
    notuse = np.zeros((kx, ky))
    for i in range(kx):
        for j in range(ky):

            if hom[i, j] == 1:
                notuse[i, j] = 3
            if MeanStrongEdge2[i, j] == 1:
                notuse[i, j] = 2
            if ((V_imUndr[i, j] == 1) or (V_imOver[i, j] == 1)):
                notuse[i, j] = 1

    for i in range(kx):
        for j in range(ky):
            if notuse[i, j] == 1:
                im[i, j] = 1

    notused = np.sum(notuse[:] != 0)
    touse = kx*ky-notused
# //////////////excl NaN

    if touse == 0:
        for i in range(kx):
            for j in range(ky):
                if hom[i, j] == 1 and im[i, j] != 1:
                    notuse[i, j] = 0

    diff_Mean_Best_scaled_temp = diff_Mean_Best_scaled.copy()
    diff_Mean_Best_scaled_tempInv = dmbsi.copy()
    for a in range(int(np.floor(blk_idx/sgrid))):
        for b in range(int(np.floor(blk_idy/sgrid))):
            if im[a, b] == 1:
                diff_Mean_Best_scaled_temp[a, b] = 0
                diff_Mean_Best_scaled_tempInv[a, b] = 1
            if diff_Mean_Best_scaled_temp[a, b] < np.mean(diff_Mean_Best_scaled) and homB == 1:
                diff_Mean_Best_scaled_temp[a, b] = 0
            if diff_Mean_Best_scaled_tempInv[a, b] < np.mean(dmbsi) and homB == 1:
                diff_Mean_Best_scaled_tempInv[a, b] = 1

    a += 1
    b += 1
    imageF = np.zeros((a, b))
    imageFInv = np.zeros((a, b))
    for x in range(a):
        for y in range(b):
            if x == 0 or x == a-1 or y == 0 or y == b-1:
                imageF[x, y] = diff_Mean_Best_scaled_temp[x, y]*(bestgrid[x, y])
            else:
                imageF[x, y] = diff_Mean_Best_scaled_temp[x, y]*(1-bestgrid[x, y])
            imageFInv[x, y] = diff_Mean_Best_scaled_tempInv[x, y]*(1-bestgrid[x, y])

    E_nofilt = imageF.copy()
    E = correlate(imageF, H, mode='constant')

    E_nofiltInv = imageFInv.copy()
    EInv = correlate(imageFInv, H, mode='constant')
    # /////////////content based filtering//////////
    uninteresting = np.zeros((touse, 1))
    uninterestingInv = np.zeros((touse, 1))
    a = -1
    for i in range(kx):
        for j in range(ky):
            if(notuse[i, j] == 0):
                a += 1
                uninteresting[a] = E[i, j]
                uninterestingInv[a] = EInv[i, j]
    MeanBlocksre = E_nofilt.copy()
    MeanBlocksreInv = E_nofiltInv.copy()
    meanuninteresting = np.mean(uninteresting)
    meanuninterestingInv = np.mean(uninterestingInv)
    for i in range(kx):
        for j in range(ky):
            if ((im[i, j] == 1) and (notuse[i, j] == 2)):
                im[i, j] = 0
            if ((notuse[i, j] == 1) or (MeanBlocksre[i, j] < meanuninteresting)):
                MeanBlocksre[i, j] = meanuninteresting
            if (((im[i, j] == 1) and (MeanBlocksre[i, j] < meanuninteresting)) or ((notuse[i, j] == 3) and (im[i, j] == 1))):
                MeanBlocksre[i, j] = meanuninteresting
            if ((notuse[i, j] == 1) or (MeanBlocksreInv[i, j] > meanuninterestingInv)):
                MeanBlocksreInv[i, j] = meanuninterestingInv
            if (((im[i, j] == 1) and (MeanBlocksreInv[i, j] > meanuninterestingInv)) or ((notuse[i, j] == 3) and (im[i, j] == 1))):
                MeanBlocksreInv[i, j] = meanuninterestingInv
    E = correlate(MeanBlocksre, H, mode='reflect')
    EInv = correlate(MeanBlocksreInv, H, mode='reflect')
    return [E, EInv]


blocksize = 6


def filtering(smap):
    """
    Fill me in please.

    Args:
        smap:

    Returns:
        meansmallAreas:
        meanbigAreas:
        meanImg:

    Todos:
        * Fill this in with proper summary
    """
    blocks = np.shape(smap)[0]
    step = int(np.sqrt(blocks))
    smallAreas = np.zeros((blocksize, blocksize))
    increment = int(step/blocksize)
    for a in range(blocksize):
        Start = int((a+1)*(blocks/blocksize)-(blocks/blocksize)+1)
        End = int((a+1)*(blocks/blocksize))
        for x in range(Start, End, step):
            for y in range(increment):
                z = x+y-1
                if (a < 3):
                    smallAreas[0, a*2] = smallAreas[0, a*2]+smap[z, 1]
                    smallAreas[0, a*2+1] = smallAreas[0, a*2+1]+smap[z+increment, 1]

                    smallAreas[1, (a*2)] = smallAreas[1, (a*2)]+smap[z+2*(increment), 1]
                    smallAreas[1, a*2+1] = smallAreas[1, a*2+1]+smap[z+3*(increment), 1]

                    smallAreas[2, a*2] = smallAreas[2, a*2]+smap[z+4*(increment), 1]
                    smallAreas[2, a*2+1] = smallAreas[2, a*2+1]+smap[z+5*(increment), 1]
                else:
                    smallAreas[3, ((a-3)*2)] = smallAreas[3, ((a-3)*2)]+smap[z, 1]
                    smallAreas[3, (a-3)*2+1] = smallAreas[3, (a-3)*2+1]+smap[z+increment, 1]

                    smallAreas[4, ((a-3)*2)] = smallAreas[4, ((a-3)*2)]+smap[z+2*(increment), 1]
                    smallAreas[4, (a-3)*2+1] = smallAreas[4, (a-3)*2+1]+smap[z+3*(increment), 1]

                    smallAreas[5, ((a-3)*2)] = smallAreas[5, ((a-3)*2)]+smap[z+4*(increment), 1]
                    smallAreas[5, (a-3)*2+1] = smallAreas[5, (a-3)*2+1]+smap[z+5*(increment), 1]
    meansmallAreas = smallAreas/100
    meanbigAreas = np.zeros((1, blocksize))
    for x in range(blocksize):
        meanbigAreas[0, x] = np.mean(meansmallAreas[x, :])
    meanImg = np.mean(meanbigAreas)
    return [meansmallAreas, meanbigAreas, meanImg]


def filteringMethod(smap, ThressSmall, ThressBigV, ThressImg):
    """
    Fill me in please.

    Args:
        smap:
        ThressSmall:
        ThressBigV:
        ThressImg:

    Returns:
        smap:

    Todos:
        * Fill this in with proper summary
    """
    blocks = np.size(smap, 0)
    step = int(np.sqrt(blocks))

    ThressBig = np.ndarray.flatten(ThressBigV)
    for x in range(blocksize):
        if ((ThressBig[x] < ThressImg) and (ThressImg < 10)):
            ThressBig[x] = ThressImg
        elif ((ThressBig[x] > ThressImg) and (ThressImg < 5)):
            ThressBig[x] = 5
        for y in range(blocksize):
            if (ThressSmall[x, y] < ThressBig[x]):
                if (ThressBig[x] < 5):
                    ThressSmall[x, y] = ThressBig[x]+1
                else:
                    ThressSmall[x, y] = ThressBig[x]
    Thresses = ThressSmall
    increment = int(step/blocksize)
    for a in range(1, blocksize+1):
        Start = int(a*(blocks/blocksize)-(blocks/blocksize))
        End = int(a*(blocks/blocksize))-1

        for x in range(Start, End, step):
            for y in range(increment):
                z = x+y
                if (a < 4):
                    if (smap[z, 1] < Thresses[0, (a*2)-2]):
                        smap[z, 0] = 59
                    if smap[z+increment, 1] < Thresses[0, a*2-1]:
                        smap[z+increment, 0] = 59
                    if smap[z+2*(increment), 1] < Thresses[1, (a*2)-2]:
                        smap[z+2*(increment), 0] = 59
                    if smap[z+3*(increment), 1] < Thresses[1, a*2-1]:
                        smap[z+3*(increment), 0] = 59
                    if smap[z+4*(increment), 1] < Thresses[2, (a*2)-2]:
                        smap[z+4*(increment), 0] = 59
                    if smap[z+5*(increment), 1] < Thresses[2, a*2-1]:
                        smap[z+5*(increment), 0] = 59
                else:
                    if smap[z, 1] < Thresses[3, ((a-3)*2)-2]:
                        smap[z, 0] = 59
                    if smap[z+increment, 1] < Thresses[3, (a-3)*2-1]:
                        smap[z+increment, 0] = 59
                    if smap[z+2*(increment), 1] < Thresses[4, ((a-3)*2)-2]:
                        smap[z+2*(increment), 0] = 59
                    if smap[z+3*(increment), 1] < Thresses[4, (a-3)*2-1]:
                        smap[z+3*(increment), 0] = 59
                    if smap[z+4*(increment), 1] < Thresses[5, ((a-3)*2)-2]:
                        smap[z+4*(increment), 0] = 59
                    if smap[z+5*(increment), 1] < Thresses[5, (a-3)*2-1]:
                        smap[z+5*(increment), 0] = 59
    return smap


def hist_adjust(arr, bins):
    """
    Fill me in please.

    Args:
        arr:
        bins:

    Returns:
        [A,B]:

    Todos:
        * Fill this in with proper summary
    """
    [A, B] = np.histogram(arr, bins)
    for i in range(1, bins):
        count = np.count_nonzero(arr == B[i])
        A[i] -= count
        A[i-1] += count
    return [A, B]


def inblockpatterns(image, bins, p, q, blk_idx, blk_idy):
    """
    Fill me in please.

    Args:
        image:
        bins:
        p:
        q:
        blk_idx:
        blk_idy:

    Returns:
        K:
        Correct:
        BlockScoreAll:

    Todos:
        * Fill this in with proper summary
    """
    Zmat = np.zeros((int(np.floor(blk_idx*blk_idy)), 2))
    a = -1
    BlockScoreAll = np.zeros((blk_idx, blk_idy))
    for i in range(blk_idx):
        Ax = (i*8)+p-1
        Ex = Ax+4
        for j in range(blk_idy):
            Ay = (j*8)+q-1
            A = image[Ax, Ay]
            B = image[Ax, Ay+1]
            C = image[Ax+1, Ay]
            D = image[Ax+1, Ay+1]

            Ey = Ay+4
            E = image[Ex, Ey]
            F = image[Ex, Ey+1]
            G = image[Ex+1, Ey]
            H = image[Ex+1, Ey+1]

            a += 1

            Zmat[a, 0] = abs(A-B-C+D)
            Zmat[a, 1] = abs(E-F-G+H)

            BlockScoreAll[i, j] = Zmat[a, 1] - Zmat[a, 0]
            if (BlockScoreAll[i, j] <= 0):
                BlockScoreAll[i, j] = 0
    norm = a
    # Currently mismatched hist fcn
    Hz = hist_adjust(Zmat[:, 0], bins)[0]
    Hzn = Hz/(norm+1)
    Hz2 = hist_adjust(Zmat[:, 1], bins)[0]
    Hz2n = Hz2/(norm+1)
    y2 = int(Hzn.size)
    K = 0
    for i in range(y2):
        K_temp = Hzn[i]-Hz2n[i]
        K += abs(K_temp)

    A = sum(Hzn[0:2])

    E = sum(Hz2n[0:2])

    if A > E:
        Correct = True
    else:
        Correct = False
    return [K, Correct, BlockScoreAll]


def predict0(Kscores):
    """
    Fill me in please.

    Args:
        Kscores:

    Returns:
        Kpredict:
        Kpre:

    Todos:
        * Fill this in with proper summary
    """
    Kpredict = np.zeros((9, 9))
    Kpredict[0:8, 0:8] = Kscores[:, :, 1]
    for i in range(8):
        Kpredict[8, i] = sum(Kpredict[:, i])
        Kpredict[i, 8] = sum(Kpredict[i, :])

    Kpre = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            Kpre[i, j] = (Kpredict[i, 8]+Kpredict[8, j])/16

    return [Kpredict, Kpre]


def predict1(Kscores, Kpredict, Kpre):
    """
    Fill me in please.

    Args:
        Kscores:
        Kpredict:
        Kpre:

    Returns:

        PossiblePoints:

    Todos:
        * Fill this in with proper summary
    """
    A = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            A[i, j] = Kscores[i, j, 0] + Kscores[i+4, j+4, 0]-Kscores[i+4, j, 0]-Kscores[i, j+4, 0]

    r1 = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
    c1 = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]

    PossiblePoints = np.zeros((len(r1), 8))
    A_point = [0, 0]
    E_point = [0, 0]

    for i in range(len(r1)):
        r = r1[i]
        c = c1[i]

        if (A[r-1, c-1] > 0):
            if (Kpredict[r-1, c-1] == 1):
                A_point[0] = r
                A_point[1] = c
                E_point[0] = r+4
                E_point[1] = c+4
            else:
                E_point[0] = r
                E_point[1] = c
                A_point[0] = r+4
                A_point[1] = c+4
        else:
            if (Kpredict[r-1, c+3] == 1):
                A_point[0] = r
                A_point[1] = c+4
                E_point[0] = r+4
                E_point[1] = c
            else:
                E_point[0] = r
                E_point[1] = c+4
                A_point[0] = r+4
                A_point[1] = c
        PossiblePoints[i, 0] = A_point[0]
        PossiblePoints[i, 1] = A_point[1]
        PossiblePoints[i, 2] = E_point[0]
        PossiblePoints[i, 3] = E_point[1]

        PossiblePoints[i, 4] = Kscores[r-1, c-1, 0]/2
        PossiblePoints[i, 5] = 0

    for i in range(len(r1)):
        PossiblePoints[i, 6] = Kpre[int(PossiblePoints[i, 0])-1, int(PossiblePoints[i, 1])-1]-Kpre[int(PossiblePoints[i, 2])-1, int(PossiblePoints[i, 3])-1]
        PossiblePoints[i, 7] = (PossiblePoints[i, 6] + PossiblePoints[i, 4])/2
    return PossiblePoints


def scores_pick_variables(BlockScoreALL, sgrid, blk_idx, blk_idy, PossiblePoints, kx, ky):
    """
    Fill me in please.

    Args:
        BlockScoreAll:
        sgrid:
        blk_idx:
        blk_idy:
        PossiblePoints:
        kx:
        ky:

    Returns:
        MeanInSpace:
        PossiblePoints:
        diff_Mean_Best_scaled:
        diff_Mean_Best_scaledInv

    Todos:
        * Fill this in with proper summary
    """

    BlockScore = np.zeros((blk_idx, blk_idy, 16))
    for i in range(16):
        p = PossiblePoints[i, 0]
        q = PossiblePoints[i, 1]
        BlockScore[:, :, i] = BlockScoreALL[:, :, int(p-1), int(q-1)]/255

    MeanInSpace = np.zeros((kx, ky, 16))
    for r in range(16):
        for i in range(kx):
            for j in range(ky):
                b = (i+1)*sgrid
                a = b-sgrid
                d = (j+1)*sgrid
                c = d-sgrid
                MeanInSpace[i, j, r] = np.mean(BlockScore[a:b, c:d, r])

    MeanOfAllGrids = np.mean(MeanInSpace, axis=2)

    BestGrid = MeanInSpace[:, :, 15]
    diff_Mean_Best = MeanOfAllGrids - BestGrid
    diff_Mean_Best_scaled = mat2gray(diff_Mean_Best)

    bg = 0
    for f in range(16):
        if (PossiblePoints[f, 0] == 4 and PossiblePoints[f, 1] == 4):
            bg = f

    for f in range(16):
        if (bg == 0):
            bg1 = np.where(PossiblePoints[:, 5] == max(PossiblePoints[:, 5]))
            bg = np.max(bg1)

    BestGridInv = np.zeros((kx, ky))
    BestGridInv = MeanInSpace[:, :, bg]
    diff_Mean_BestInv = MeanOfAllGrids - BestGridInv
    diff_Mean_Best_scaledInv = mat2gray(diff_Mean_BestInv)

    return [MeanInSpace, PossiblePoints, diff_Mean_Best_scaled, diff_Mean_Best_scaledInv]


def CAGI(impath):
    """
    Main driver for CAGI algorithm.

    Args:
        impath:

    Returns:
        Result_CAGI: Equivalent to OutputMap
        Result_Inv_CAGI: Other output of CAGI.
    """
    # Read image in as double RGB
    BGR = cv2.imread(impath)
    RGB = np.double(BGR[..., ::-1])

    (height, width, color) = RGB.shape
    V_im = cv2.cvtColor(np.uint8(RGB), cv2.COLOR_RGB2HSV)[:, :, 2]

    # Store the pixels for Y of YCbCr
    pixels = 16 / 255 + (0.256788 * RGB[:, :, 0] + 0.504129 * RGB[:, :, 1] + 0.0979058 * RGB[:, :, 2])

    if ((height*width) < (480*640)):
        sgrid = 2
    else:
        sgrid = 3
    bins = 40
    imageGS = pixels
    (x, y) = imageGS.shape
    blk_idx = int(np.floor((x/8)-1))
    blk_idy = int(np.floor((y/8)-1))
    kx = int(np.floor(blk_idx/sgrid))
    ky = int(np.floor(blk_idy/sgrid))
    BlockScoreAll = np.zeros((blk_idx, blk_idy, 8, 8))
    Kscores = np.zeros((8, 8, 2))
    for p in range(8):
        for q in range(8):
            (K, Correct, BlockScoreAll[:, :, p, q]) = inblockpatterns(imageGS, bins, p+1, q+1, blk_idx, blk_idy)
            if (K > 1.999999):
                Kscores[p, q, 0] = 0
            else:
                Kscores[p, q, 0] = K
            Kscores[p, q, 1] = Correct

    [Kpredict, Kpre] = predict0(Kscores)
    PossiblePoints = predict1(Kscores, Kpredict, Kpre)
    PossiblePoints = PossiblePoints[np.argsort(PossiblePoints[:, 6])]
    [MeanContent, MeanStrongEdge] = MainTrain(RGB, blk_idx, blk_idy)
    MeanContent2 = np.zeros((kx, ky))
    for i in range(kx):
        for j in range(ky):
            a = i*sgrid
            b = j*sgrid
            ccc = sgrid
            MeanContent2[i, j] = np.mean(MeanContent[a:a+ccc, b:b+ccc])
    [MeanInSpace, PossiblePoints, diff_Mean_Best_scaled, dmbsi] = scores_pick_variables(BlockScoreAll, sgrid, blk_idx, blk_idy, PossiblePoints, kx, ky)
    [E, EInv] = characterizeblocks(MeanContent2, MeanStrongEdge, V_im, blk_idx, blk_idy, MeanInSpace, diff_Mean_Best_scaled, dmbsi, sgrid, PossiblePoints, kx, ky)
    Result_CAGI = RescaleToImageResult(E, sgrid, kx, ky, pixels)
    Result_Inv_CAGI = RescaleToImageResult(EInv, sgrid, kx, ky, pixels)
    return [Result_CAGI, Result_Inv_CAGI]


def getMasks():

    """
    Return image masks.

    Args:

    Returns:
        PMasks:
        MMasks:
        MaskWhite:
    """

    PMasks = np.load(os.path.join(os.path.dirname(__file__), 'PMasks.npy'))
    MMasks = np.load(os.path.join(os.path.dirname(__file__), 'MMasks.npy'))
    MaskWhite = np.array([[10],
                         [30],
                         [50],
                         [70],
                         [90],
                         [20],
                         [40],
                         [60],
                         [80],
                         [12],
                         [30],
                         [50],
                         [70],
                         [88],
                         [15],
                         [28],
                         [45],
                         [64],
                         [79],
                         [85],
                         [12],
                         [30],
                         [50],
                         [70],
                         [88],
                         [20],
                         [40],
                         [60],
                         [80],
                         [10],
                         [30],
                         [50],
                         [70],
                         [90],
                         [20],
                         [40],
                         [60],
                         [80],
                         [12],
                         [30],
                         [50],
                         [70],
                         [88],
                         [15],
                         [21],
                         [36],
                         [55],
                         [72],
                         [85],
                         [12],
                         [30],
                         [50],
                         [70],
                         [88],
                         [20],
                         [40],
                         [60],
                         [80]], dtype=np.uint8)

    return [PMasks, MMasks, MaskWhite]
