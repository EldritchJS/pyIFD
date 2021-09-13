from pyIFD.ADQ1 import detectDQ
from pyIFD.ADQ2 import getJmap
from pyIFD.ADQ3 import BenfordDQ
from pyIFD.BLK import GetBlockGrid
from pyIFD.CAGI import CAGI
from pyIFD.CFA1 import CFA1
from pyIFD.CFA2 import CFA2
from pyIFD.DCT import DCT
from pyIFD.ELA import ELA
from pyIFD.GHOST import GHOST
from pyIFD.NADQ import NADQ
from pyIFD.NOI1 import GetNoiseMap
from pyIFD.NOI2 import GetNoiseMaps 
from pyIFD.NOI4 import MedFiltForensics
from pyIFD.NOI5 import PCANoise
import numpy as np
import scipy.io as spio
from skimage.metrics import structural_similarity as comp
import sys
import os
import argparse
import logging

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
        if infilename[-4:] != ".jpg":
            print("ADQ2 only takes .jpg inputs")
            return 1
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
        if infilename[-4:] != ".jpg":
            print("ADQ3 only takes .jpg inputs")
            return 1
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

    elif algoname == 'CFA1':
        cfa1test = CFA1(infilename)
        cfa1mat = spio.loadmat(matfilename)
        sim = 0

        try:
            sim = comp(cfa1mat['OutputMap'], cfa1test)
        except ValueError as e:
            print(e)
            return retVal

        if(sim < criteria):
            print('CFA1: FAIL Similarity: ' + str(sim))
        else:
            print('CFA1: PASS')
            retVal = True


    elif algoname == 'CFA2':
        cfa2test = CFA2(infilename)
        cfa2mat = spio.loadmat(matfilename)
        sim = 0

        try:
            sim = comp(cfa2mat['OutputMap'], cfa2test)
        except ValueError as e:
            print(e)
            return retVal

        if(sim < criteria):
            print('CFA2: FAIL Similarity: ' + str(sim))
        else:
            print('CFA2: PASS')
            retVal = True

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

    elif algoname == 'NADQ':
        if infilename[-4:] != ".jpg":
            print("NADQ only takes .jpg inputs")
            return 1
        nadqtest = NADQ(infilename)
        nadqmat = spio.loadmat(matfilename)
        sim = 0

        try:
            sim = comp(nadqmat['OutputMap'], nadqtest)
        except ValueError as e:
            print(e)
            return retVal

        if(sim < criteria):
            print('NADQ: FAIL Similarity: ' + str(sim))
        else:
            print('NADQ: PASS')
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


def main(args):
    if args.rootdircorrect is True:
        for root, dirs, files in os.walk(args.imagefilesrootdir):
            dirs.sort()
            for basefilename in sorted(files):
                imagefilename = os.path.join(root,basefilename)
                splitimage = os.path.splitext(basefilename)
                if(splitimage[1] == '.jpg'):
                    matfiledir = args.groundtruthfilesrootdir + '/' + splitimage[0]
                    for algorithm in args.algorithms:
                        matfilename = matfiledir + '/' + splitimage[0] + '_' + algorithm + '.mat'
                        print('Validating image ' + basefilename + ' for algorithm ' + algorithm)
                        validate_algo(imagefilename, matfilename, algorithm)
    elif args.singlefilecorrect is True:
        basefilename = os.path.splitext(os.path.realpath(args.imagefilename))[0].split('_')[0]
        for algorithm in args.algorithms:
            print('Validating image ' + args.imagefilename + ' for algorithm ' + algorithm)
            groundtruthfilename = basefilename + '_' + algorithm + '.mat'
            validate_algo(args.imagefilename, groundtruthfilename, algorithm, args.simcriteria)


def get_arg(env, default):
    return os.getenv(env) if os.getenv(env, "") != "" else default


def parse_args(parser):
    args = parser.parse_args()
    args.algorithms = get_arg('PYIFD_ALGORITHMS', args.algorithms).split(',')
    args.imagefilename = get_arg('PYIFD_IMAGE_FILENAME', args.imagefilename)
    args.imagefilesrootdir = get_arg('PYIFD_IMAGE_ROOTDIR', args.imagefilesrootdir)
    args.groundtruthfilesrootdir = get_arg('PYIFD_GROUND_TRUTH_ROOTDIR', args.groundtruthfilesrootdir)
    args.simcriteria = float(get_arg('PYIFD_SIM_CRITERIA', args.simcriteria))
    
    args.singlefilecorrect = args.imagefilename is not None
    args.rootdircorrect = (args.imagefilesrootdir is not None) and (args.groundtruthfilesrootdir is not None)
    
    if args.singlefilecorrect and args.rootdircorrect:
        logging.warning('Both single file and image/ground truth rootdirs defined. Defaulting to rootdirs')
    elif (args.singlefilecorrect or args.rootdircorrect) is not True:
        logging.error('Either imagefilename must be defined or imagefilesrootdir and groundtruthfilesrootdir must be defined')
        args = None

    return args


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info('Starting pyIFD validation')
    parser = argparse.ArgumentParser(description='Get algorithm list, image filename/root dir, ground truth filename/root dir, for each algorithm process each image and compare with ground truth')
    parser.add_argument(
        '--algorithms',
        help='Comma separated list of algorithms to run, env variable PYIFD_ALGORITHMS',
        default='All')

    parser.add_argument(
        '--imagefilename',
        help='Input image filename, env variable PYIFD_IMAGE_FILENAME',
        default=None)

    parser.add_argument(
        '--groundtruthfilename',
        help='Input image ground truth filename, env variable PYIFD_GROUND_TRUTH_FILENAME',
        default=None)
            
    parser.add_argument(
        '--imagefilesrootdir',
        help='Input images root dir which will be searched for images, processing each, env variable PYIFD_IMAGE_ROOTDIR',
        default=None)

    parser.add_argument(
        '--groundtruthfilesrootdir',
        help='Input image ground truth root dir, env variable PYIFD_GROUND_TRUTH_ROOTDIR',
        default=None)

    parser.add_argument(
        '--simcriteria',
        help='Algorithm similarity criteria, env variable PYIFD_SIM_CRITERIA',
        default=0.99)

    cmdline_args = parse_args(parser)
    if cmdline_args is not None:
        logging.info('Starting validation')
        main(cmdline_args)
    logging.info('Exiting validation')
