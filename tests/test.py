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

ADQ1_CRITERIA = 0.99
ADQ2_CRITERIA = 0.99
ADQ3_CRITERIA = 0.99
BLK_CRITERIA = 0.99
CAGI_CRITERIA = 0.99
ELA_CRITERIA = 0.99
GHOST_CRITERIA = 0.99
NOI1_CRITERIA = 0.99
NOI2_CRITERIA = 0.99
NOI4_CRITERIA = 0.99
NOI5_CRITERIA = 0.99

adq1test=detectDQ('../data/demo_adq1.jpg')
adqmat=spio.loadmat('../data/demo_adq1_jpg.mat')
if(comp(adqmat['OutputMap'],adq1test[0])<ADQ1_CRITERIA):
    print('ADQ1 JPEG: FAIL')
else:
    print('ADQ1 JPEG: PASS')

adq1test=detectDQ('../data/demo_adq1.png')
adq1mat=spio.loadmat('../data/demo_adq1_png.mat')
if(comp(adq1mat['OutputMap'],adq1test[0])<ADQ1_CRITERIA):
    print('ADQ1 PNG: FAIL')
else:
    print('ADQ1 PNG: PASS')    

adq2test=getJmap('../data/demo_adq2.jpg')
adq2mat=spio.loadmat('../data/demo_adq2.mat')
if(comp(adq2mat['OutputMap'],adq2test[0])<ADQ2_CRITERIA):
    print('ADQ2: FAIL')
else:
    print('ADQ2: PASS')

adq3test=BenfordDQ('../data/demo_adq3.jpg')
adq3mat=spio.loadmat('../data/demo_adq3.mat')
if(comp(adq3mat['OutputMap'],adq3test)<ADQ3_CRITERIA):
    print('ADQ3: FAIL')
else:
    print('ADQ3: PASS')

blktest=GetBlockGrid('../data/demo_blk.jpg')
blkmat=spio.loadmat('../data/demo_blk.mat')
if(comp(blkmat['OutputMap'],blktest[0])<BLK_CRITERIA):
    print('BLK: FAIL')
else:
    print('BLK: PASS')

cagitest=CAGI('../data/demo_cagi.jpg')
cagimat=spio.loadmat('../data/demo_cagi.mat')
if(comp(cagimat['a'],cagitest[0])<CAGI_CRITERIA):
    print('CAGI: FAIL')
else:
    print('CAGI: PASS')

if(comp(cagimat['b'],cagitest[1])<CAGI_CRITERIA):
    print('CAGI INVERSE: FAIL')
else:
    print('CAGI INVERSE: PASS')

elatest=ELA('../data/demo_ela.jpg')
elamat=spio.loadmat('../data/demo_ela.mat')
if(comp(elamat['OutputMap'],elatest.astype('uint16'))<ELA_CRITERIA):
    print('ELA: FAIL')
else:
    print('ELA: PASS')

ghosttest=GHOST('../data/demo_ghost.jpg')
ghostmat=spio.loadmat('../data/demo_ghost.mat')
matDispImages = ghostmat['dispImages'][0]
pyDispImages = ghosttest[2]
similarity=[]
from skimage.metrics import structural_similarity as comp
for i in range(len(matDispImages)):
    similarity.append(comp(matDispImages[i],pyDispImages[i]))
if(np.mean(similarity)<GHOST_CRITERIA):
    print('GHOST: FAIL')
else:
    print('GHOST: PASS')

noi1test=GetNoiseMap('../data/demo_noi1.tif')
noi1mat=spio.loadmat('../data/demo_noi1.mat')
if(comp(noi1mat['Map'],noi1test)<NOI1_CRITERIA):
    print('NOI1: FAIL')
else:
    print('NOI1: PASS')

noi2test=GetNoiseMaps('../data/demo_noi2.tif')
noi2mat=spio.loadmat('../data/demo_noi2.mat')
if(comp(noi2mat['OutputMap'],noi2test)<NOI2_CRITERIA):
    print('NOI2: FAIL')
else:
    print('NOI2: PASS')


noi4test=MedFiltForensics('../data/demo_noi4.tif')
noi4mat=spio.loadmat('../data/demo_noi4.mat')
if(comp(noi4mat['OutputMap'],noi4test)<NOI4_CRITERIA):
    print('NOI4: FAIL')
else:
    print('NOI4: PASS')

noi5test=PCANoise('../data/demo_noi5.tif')
noi5mat=spio.loadmat('../data/demo_noi5.mat')
if(comp(noi5mat['Noise_mix2'],noi5test[0])<NOI5_CRITERIA):
    print('NOI5 NOISE_MIX2: FAIL')
else:
    print('NOI5 NOISE_MIX2: PASS')
if(comp(noi5mat['highlighted'],noi5test[1],multichannel=True)<NOI5_CRITERIA):
    print('NOI5 HIGHLIGHTED: FAIL')
else:
    print('NOI5 HIGHLIGHTED: PASS')

