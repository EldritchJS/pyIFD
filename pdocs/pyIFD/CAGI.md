Module pyIFD.CAGI
=================
This module provides the CAGI algorithm

Functions
---------

    
`CAGI(impath)`
:   Main driver for CAGI algorithm.
    
    Args:
        impath:
    
    Returns:
        Result_CAGI: Equivalent to OutputMap
        Result_Inv_CAGI: Other output of CAGI.

    
`ImageTiling(OImg)`
:   Fill me in please.
    
    Args:
        OImg:
    
    Returns:
        tile:
    
    Todos:
        * Fill this in with proper summary

    
`MainTrain(R10, blk_idx, blk_idy)`
:   Fill me in please.
    
    Args:
        R10:
        blk_idx:
        blk_idy:
    
    Returns:
        MeanContent:
        MeanStrongEdge:
    
    Todos:
        * Fill this in with proper summary

    
`PaintimgEdges(smap, MMasks, scale)`
:   Fill me in please.
    
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

    
`RescaleToImageResult(E, sgrid, kx, ky, pixels)`
:   Fill me in please.
    
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

    
`SmapIng(ImgTiles, MaskTiles, WhiteMaskPoints)`
:   Fill me in please.
    
    Args:
        ImgTiles:
        MaskTiles:
        WhiteMaskPoints:
    
    Returns:
        smap:
    
    Todos:
        * Fill this in with proper summary

    
`characterizeblocks(MeanContent2, MeanStrongEdge, V_im, blk_idx, blk_idy, MeanInSpace, diff_Mean_Best_scaled, dmbsi, sgrid, PossiblePoints, kx, ky)`
:   Fill me in please.
    
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

    
`filtering(smap)`
:   Fill me in please.
    
    Args:
        smap:
    
    Returns:
        meansmallAreas:
        meanbigAreas:
        meanImg:
    
    Todos:
        * Fill this in with proper summary

    
`filteringMethod(smap, ThressSmall, ThressBigV, ThressImg)`
:   Fill me in please.
    
    Args:
        smap:
        ThressSmall:
        ThressBigV:
        ThressImg:
    
    Returns:
        smap:
    
    Todos:
        * Fill this in with proper summary

    
`getMasks()`
:   Return image masks.
    
    Args:
    
    Returns:
        PMasks:
        MMasks:
        MaskWhite:

    
`hist_adjust(arr, bins)`
:   Fill me in please.
    
    Args:
        arr:
        bins:
    
    Returns:
        [A,B]:
    
    Todos:
        * Fill this in with proper summary

    
`im2double(im)`
:   Converts image to type double.
    
    Args:
        im: Input image
    
    Returns:
        image as double: Converts type of im to double. Scales so elements lie from 0 to 1.

    
`inblockpatterns(image, bins, p, q, blk_idx, blk_idy)`
:   Fill me in please.
    
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

    
`mat2gray(A)`
:   Converts matrix to have values from 0-1.
    
    Args:
        A: Input matrix.
    
    Returns:
        Gray matrix with values from 0-1.
    
    Todos:
        * Fill this in with proper summary

    
`predict0(Kscores)`
:   Fill me in please.
    
    Args:
        Kscores:
    
    Returns:
        Kpredict:
        Kpre:
    
    Todos:
        * Fill this in with proper summary

    
`predict1(Kscores, Kpredict, Kpre)`
:   Fill me in please.
    
    Args:
        Kscores:
        Kpredict:
        Kpre:
    
    Returns:
    
        PossiblePoints:
    
    Todos:
        * Fill this in with proper summary

    
`scores_pick_variables(BlockScoreALL, sgrid, blk_idx, blk_idy, PossiblePoints, kx, ky)`
:   Fill me in please.
    
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