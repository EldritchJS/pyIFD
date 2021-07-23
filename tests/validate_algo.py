from pyIFD.ADQ1 import detectDQ
from pyIFD.ADQ2 import getJmap
from pyIFD.ADQ3 import BenfordDQ
from pyIFD.BLK import GetBlockGrid
from pyIFD.CAGI import CAGI
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

ADQ1_CRITERIA = 0.99
ADQ2_CRITERIA = 0.99
ADQ3_CRITERIA = 0.99
BLK_CRITERIA = 0.90
CAGI_CRITERIA = 0.90
ELA_CRITERIA = 0.99
GHOST_CRITERIA = 0.99
NOI1_CRITERIA = 0.90
NOI2_CRITERIA = 0.90
NOI4_CRITERIA = 0.90
NOI5_CRITERIA = 0.90

def validate_algo(infilename, matfilename, algoname):

    if algoname == 'ADQ1':
        adq1test=detectDQ(infilename)
        adq1mat=spio.loadmat(matfilename)
        sim = 0

        try:
            sim = comp(adq1mat['OutputMap'],adq1test[0])
        except ValueError as e:
            print(e)
        
        if(sim<ADQ1_CRITERIA):
            print('ADQ1: FAIL Similarity: ' + str(sim))
        else:
            print('ADQ1: PASS')

    elif algoname == 'ADQ2':
        adq2test=getJmap(infilename)
        adq2mat=spio.loadmat(matfilename)
        sim = 0

        try:
            sim = comp(adq2mat['OutputMap'],adq2test[0])
        except ValueError as e:
            print(e)
        
        if(sim<ADQ2_CRITERIA):
            print('ADQ2: FAIL Similarity: ' + str(sim))
        else:
            print('ADQ2: PASS')            

    elif algoname == 'ADQ3':
        adq3test=BenfordDQ(infilename)
        adq3mat=spio.loadmat(matfilename)
        sim = 0

        try:
            sim = comp(adq3mat['OutputMap'],adq3test)
        except ValueError as e:
            print(e)

        if(sim<ADQ3_CRITERIA):
            print('ADQ3: FAIL Similarity: ' + str(sim))
        else:
            print('ADQ3: PASS')

    elif algoname == 'BLK':
        blktest=GetBlockGrid(infilename)
        blkmat=spio.loadmat(matfilename)
        sim = 0

        try:
            sim = comp(blkmat['OutputMap'],blktest[0])
        except ValueError as e:
            print(e)
        
        if(sim<BLK_CRITERIA):
            print('BLK: FAIL Similarity: ' + str(sim))
        else:
            print('BLK: PASS')

    elif algoname == 'CAGI':    
        cagitest=CAGI(infilename)
        cagimat=spio.loadmat(matfilename)
        sim = 0

        try:
            sim = comp(cagimat['OutputMap'],cagitest[0])
        except ValueError as e:
            print(e)

        if(sim<CAGI_CRITERIA):
            print('CAGI: FAIL Similarity: ' + str(sim))
        else:
            print('CAGI: PASS')

        sim = 0

        try:
            sim = comp(cagimat['OutputMap_Inverse'],cagitest[1])
        except ValueError as e:
            print(e)

        if(sim<CAGI_CRITERIA):
            print('CAGI INVERSE: FAIL Similarity: ' + str(sim))
        else:
            print('CAGI INVERSE: PASS')
    
    elif algoname == 'ELA':
        elatest=ELA(infilename)
        elamat=spio.loadmat(matfilename)
        sim = 0

        try:
            sim=comp(elamat['OutputMap'],elatest.astype(np.uint8))
        except ValueError as e:
            print(e)

        if(sim<ELA_CRITERIA):
            print('ELA: FAIL Similarity: ' + str(sim))
        else:
            print('ELA: PASS')

    elif algoname == 'GHO':
        ghosttest=GHOST(infilename)
        ghostmat=spio.loadmat(matfilename)
        matDispImages = ghostmat['OutputMap'][0]
        pyDispImages = ghosttest[2]
        similarity=[]
        for i in range(len(matDispImages)):
            sim = 0
            try:
                sim = comp(matDispImages[i],pyDispImages[i])
            except ValueError as e:
                print(e)

            similarity.append(sim)
        sim = np.mean(similarity)
        if(sim<GHOST_CRITERIA):
            print('GHOST: FAIL Similarity: ' + str(sim))
        else:
            print('GHOST: PASS')

    elif algoname == 'NOI1':
        noi1test=GetNoiseMap(infilename)
        noi1mat=spio.loadmat(matfilename)
        sim = 0

        try:
            sim = comp(noi1mat['OutputMap'],noi1test)
        except ValueError as e:
            print(e)

        if(sim<NOI1_CRITERIA):
            print('NOI1: FAIL Similarity: ' + str(sim))
        else:
            print('NOI1: PASS')

    elif algoname == 'NOI2':
        noi2test=GetNoiseMaps(infilename,filter_type='haar')
        noi2mat=spio.loadmat(matfilename)
        sim = 0

        try:
            sim = comp(noi2mat['OutputMap'],noi2test)
        except ValueError as e:
            print(e)

        if(sim<NOI2_CRITERIA):
            print('NOI2: FAIL Similarity: ' + str(sim))
        else:
            print('NOI2: PASS')

    elif algoname == 'NOI4':
        noi4test=MedFiltForensics(infilename, Flatten=False)
        noi4mat=spio.loadmat(matfilename)
        sim = 0

        try:
            sim = comp(noi4mat['OutputMap'],noi4test,multichannel=True)
        except ValueError as e:
            print(e)

        if(sim<NOI4_CRITERIA):
            print('NOI4: FAIL Similarity: ' + str(sim))
        else:
            print('NOI4: PASS')

    elif algoname == 'NOI5':
        try:
            noi5test=PCANoise(infilename)
        except:
            print('NOI5: ALGO FAILED')
            return
        noi5mat=spio.loadmat(matfilename)
        sim = 0

        try:
            sim = comp(noi5mat['OutputMap'],noi5test[0])
        except ValueError as e:
            print(e)

        if(sim<NOI5_CRITERIA):
            print('NOI5 OutputMap: FAIL Similarity: ' + str(sim))
        else:
            print('NOI5 OutputMap: PASS')

        sim = 0
        try:
            sim = comp(noi5mat['OutputMap_Quant'],noi5test[1],multichannel=True)
        except ValueError as e:
            print(e)

        if(sim<NOI5_CRITERIA):
            print('NOI5 OutputMap_Quant: FAIL Similarity: ' + str(sim))
        else:
            print('NOI5 OutputMap_Quant: PASS')

    else:
        print('Unknown algorithm: ' + algoname)

#algorithms = ['ADQ1', 'ADQ2', 'ADQ3', 'BLK', 'CAGI', 'ELA', 'GHO', 'NOI1', 'NOI2', 'NOI4', 'NOI5']
algorithms = ['NOI5']

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
