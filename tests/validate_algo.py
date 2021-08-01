from pyIFD.ADQ1 import detectDQ
from pyIFD.ADQ2 import getJmap
from pyIFD.ADQ3 import BenfordDQ
from pyIFD.BLK import GetBlockGrid
from pyIFD.CAGI import CAGI
from pyIFD.DCT import DCT
from pyIFD.ELA import ELA
from pyIFD.GHOST import GHOST
from pyIFD.NOI1 import GetNoiseMap
from pyIFD.NOI2 import GetNoiseMaps 
from pyIFD.NOI4 import MedFiltForensics
from pyIFD.NOI5 import PCANoise
import numpy as np
import scipy.io as spio
from skimage.metrics import structural_similarity as comp
import sys
import os


def validate_algo(infilename, matfilename, algoname, criteria=0.99):

    retVal = False

    if algoname == 'ADQ1':
        adq1test = detectDQ(infilename)
        adq1mat = spio.loadmat(matfilename)
        sim = 0

        try:
            sim = comp(adq1mat['OutputMap'], adq1test[0])
        except ValueError as e:
            print(e)
            return retVal
        
        if(sim < criteria):
            print('ADQ1: FAIL Similarity: ' + str(sim))
        else:
            print('ADQ1: PASS')
            retVal = True

    elif algoname == 'ADQ2':
        adq2test = getJmap(infilename)
        adq2mat = spio.loadmat(matfilename)
        sim = 0

        try:
            sim = comp(adq2mat['OutputMap'], adq2test[0])
        except ValueError as e:
            print(e)
            return retVal
        
        if(sim < criteria):
            print('ADQ2: FAIL Similarity: ' + str(sim))
        else:
            print('ADQ2: PASS')            
            retVal = True

    elif algoname == 'ADQ3':
        adq3test = BenfordDQ(infilename)
        adq3mat = spio.loadmat(matfilename)
        sim = 0

        try:
            sim = comp(adq3mat['OutputMap'], adq3test)
        except ValueError as e:
            print(e)
            return retVal

        if(sim < criteria):
            print('ADQ3: FAIL Similarity: ' + str(sim))
        else:
            print('ADQ3: PASS')
            retVal = True

    elif algoname == 'BLK':
        blktest = GetBlockGrid(infilename)
        blkmat = spio.loadmat(matfilename)
        sim = 0

        try:
            sim = comp(blkmat['OutputMap'], blktest[0])
        except ValueError as e:
            print(e)
            return retVal
        
        if(sim < criteria):
            print('BLK: FAIL Similarity: ' + str(sim))
        else:
            print('BLK: PASS')
            retVal = True

    elif algoname == 'CAGI':    
        cagitest = CAGI(infilename)
        cagimat = spio.loadmat(matfilename)
        sim = 0

        try:
            sim = comp(cagimat['OutputMap'], cagitest[0])
        except ValueError as e:
            print(e)
            return retVal

        if(sim < criteria):
            print('CAGI: FAIL Similarity: ' + str(sim))
        else:
            print('CAGI: PASS')
            retVal = True

        sim = 0

        try:
            sim = comp(cagimat['OutputMap_Inverse'], cagitest[1])
        except ValueError as e:
            print(e)
            return retVal

        if(sim < criteria):
            print('CAGI INVERSE: FAIL Similarity: ' + str(sim))
            retVal = False
        else:
            print('CAGI INVERSE: PASS')

    elif algoname == 'DCT':
        dcttest = DCT(infilename)
        dctmat = spio.loadmat(matfilename)
        sim = 0

        try:
            sim = comp(dctmat['OutputMap'], dcttest)
        except ValueError as e:
            print(e)
            return retVal

        if(sim < criteria):
            print('DCT: FAIL Similarity: ' + str(sim))
        else:
            print('DCT: PASS')
            retVal = True

    elif algoname == 'ELA':
        elatest = ELA(infilename)
        elamat = spio.loadmat(matfilename)
        sim = 0

        try:
            sim = comp(elamat['OutputMap'], elatest.astype(np.uint8))
        except ValueError as e:
            print(e)
            return retVal

        if(sim < criteria):
            print('ELA: FAIL Similarity: ' + str(sim))
        else:
            print('ELA: PASS')
            retVal = True

    elif algoname == 'GHO':
        ghosttest = GHOST(infilename)
        ghostmat = spio.loadmat(matfilename)
        matDispImages = ghostmat['OutputMap'][0]
        pyDispImages = ghosttest[2]
        similarity = []
        for i in range(len(matDispImages)):
            sim = 0
            try:
                sim = comp(matDispImages[i], pyDispImages[i])
            except ValueError as e:
                print(e)
                return retVal

            similarity.append(sim)
        sim = np.mean(similarity)
        if(sim < criteria):
            print('GHOST: FAIL Similarity: ' + str(sim))
        else:
            print('GHOST: PASS')
            retVal = True

    elif algoname == 'NOI1':
        noi1test = GetNoiseMap(infilename)
        noi1mat = spio.loadmat(matfilename)
        sim = 0

        try:
            sim = comp(noi1mat['OutputMap'], noi1test)
        except ValueError as e:
            print(e)
            return retVal

        if(sim < criteria):
            print('NOI1: FAIL Similarity: ' + str(sim))
        else:
            print('NOI1: PASS')
            retVal = True

    elif algoname == 'NOI2':
        noi2test = GetNoiseMaps(infilename, filter_type='haar')
        noi2mat = spio.loadmat(matfilename)
        sim = 0

        try:
            sim = comp(noi2mat['OutputMap'], noi2test)
        except ValueError as e:
            print(e)
            return retVal

        if(sim < criteria):
            print('NOI2: FAIL Similarity: ' + str(sim))
        else:
            print('NOI2: PASS')
            retVal = True

    elif algoname == 'NOI4':
        noi4test = MedFiltForensics(infilename, Flatten=False)
        noi4mat = spio.loadmat(matfilename)
        sim = 0

        try:
            sim = comp(noi4mat['OutputMap'], noi4test, multichannel=True)
        except ValueError as e:
            print(e)
            return retVal

        if(sim < criteria):
            print('NOI4: FAIL Similarity: ' + str(sim))
        else:
            print('NOI4: PASS')
            retVal = True

    elif algoname == 'NOI5':
        try:
            noi5test = PCANoise(infilename)
        except:
            print('NOI5: ALGO FAILED')
            return retVal

        noi5mat = spio.loadmat(matfilename)
        sim = 0

        try:
            sim = comp(noi5mat['OutputMap'], noi5test[0])
        except ValueError as e:
            print(e)
            return retVal

        if(sim < criteria):
            print('NOI5 OutputMap: FAIL Similarity: ' + str(sim))
        else:
            print('NOI5 OutputMap: PASS')
            retVal = True

    else:
        print('Unknown algorithm: ' + algoname)

    return retVal
#algorithms = ['ADQ1', 'ADQ2', 'ADQ3', 'BLK', 'CAGI', 'DCT', 'ELA', 'GHO', 'NOI1', 'NOI2', 'NOI4', 'NOI5']
algorithms = ['DCT']


def main(argv):
    for root, dirs, files in os.walk(sys.argv[1]):
        dirs.sort()
        for basefilename in sorted(files):
            imagefilename = os.path.join(root,basefilename)
            splitimage = os.path.splitext(basefilename)
            if(splitimage[1] == '.jpg'):
                matfiledir = sys.argv[2] + '/' + splitimage[0]
                for algorithm in algorithms:
                    matfilename = matfiledir + '/' + splitimage[0] + '_' + algorithm + '.mat'
                    print('Validating image ' + basefilename + ' for algorithm ' + algorithm)
                    validate_algo(imagefilename, matfilename, algorithm)

# Usage: validate_algo.py <IMAGE FILE BASE DIRECTORY> <MATLAB FILE BASE DIRECTORY> 

if __name__ == "__main__":
    main(sys.argv[1:])
