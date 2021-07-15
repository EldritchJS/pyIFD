###ADQ1###

Validating image 100.jpg for algorithm ADQ1
ADQ1: PASS
Validating image 101.jpg for algorithm ADQ1
/home/jason/.virtualenvs/ifd/lib/python3.8/site-packages/numpy/core/fromnumeric.py:3723: RuntimeWarning: Degrees of freedom <= 0 for slice
  return _methods._var(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
/home/jason/.virtualenvs/ifd/lib/python3.8/site-packages/numpy/core/_methods.py:222: RuntimeWarning: invalid value encountered in true_divide
  arrmean = um.true_divide(arrmean, div, out=arrmean, casting='unsafe',
/home/jason/.virtualenvs/ifd/lib/python3.8/site-packages/numpy/core/_methods.py:254: RuntimeWarning: invalid value encountered in double_scalars
  ret = ret.dtype.type(ret / rcount)
ADQ1: PASS

###BLK###


  File "validate_algo.py", line 64, in main
    sim = comp(blkmat['OutputMap'],blktest[0])
  File "/home/jason/.virtualenvs/ifd/lib/python3.8/site-packages/skimage/metrics/_structural_similarity.py", line 89, in structural_similarity
    check_shape_equality(im1, im2)
  File "/home/jason/.virtualenvs/ifd/lib/python3.8/site-packages/skimage/_shared/utils.py", line 294, in check_shape_equality
    raise ValueError('Input images must have the same dimensions.')
ValueError: Input images must have the same dimensions.


Of those that don't get the dimensions error, image 176 has similarity 0.77. All others 0.9 or higher

###CAGI###

Similarties below 0.9

Processing ../../image-manipulation-detectors/datasmall/109/109.jpg
CAGI: FAIL Similarity: 0.43017219515101224

Processing ../../image-manipulation-detectors/datasmall/117/117.jpg
CAGI: FAIL Similarity: 0.7266555219103239

Processing ../../image-manipulation-detectors/datasmall/132/132.jpg
CAGI: FAIL Similarity: 0.3890302307955631

Processing ../../image-manipulation-detectors/datasmall/137/137.jpg
CAGI: FAIL Similarity: 0.2471761004128107

Processing ../../image-manipulation-detectors/datasmall/153/153.jpg
CAGI: FAIL Similarity: 0.8719571372432249

Processing ../../image-manipulation-detectors/datasmall/156/156.jpg
CAGI: FAIL Similarity: 0.8228678813097191
CAGI INVERSE: FAIL Similarity: 0.8544103279280683

Processing ../../image-manipulation-detectors/datasmall/170/170.jpg
CAGI: FAIL Similarity: 0.8541673646451104

Processing ../../image-manipulation-detectors/datasmall/171/171.jpg
CAGI: FAIL Similarity: 0.6957841892324509

Processing ../../image-manipulation-detectors/datasmall/173/173.jpg
CAGI: FAIL Similarity: 0.4402535442719123

Processing ../../image-manipulation-detectors/datasmall/176/176.jpg
CAGI: FAIL Similarity: 0.8041531513428627

Processing ../../image-manipulation-detectors/datasmall/190/190.jpg
CAGI: FAIL Similarity: 0.7658262586649569

Processing ../../image-manipulation-detectors/datasmall/193/193.jpg
CAGI: FAIL Similarity: 0.8361149631724648

Processing ../../image-manipulation-detectors/datasmall/194/194.jpg
CAGI: FAIL Similarity: 0.7717142493634741

Processing ../../image-manipulation-detectors/datasmall/197/197.jpg
CAGI: FAIL Similarity: 0.8709206227659007

###GHO###

Images 144, 163, 170, 187, 197

 File "validate_algo.py", line 97, in main   
 similarity.append(comp(matDispImages[i],pyDispImages[i]))
  File "/home/jason/.virtualenvs/ifd/lib/python3.8/site-packages/skimage/metrics/_structural_similarity.py", line 89, in structural_similarity
    check_shape_equality(im1, im2)
  File "/home/jason/.virtualenvs/ifd/lib/python3.8/site-packages/skimage/_shared/utils.py", line 294, in check_shape_equality
    raise ValueError('Input images must have the same dimensions.')
ValueError: Input images must have the same dimensions.

###NOI1###

Image 107 0.88 similarity

###NOI2###

Image 103 0.34 similarity
Image 138 0.299 similarity

`zero encountered in true_divide
  noiK = (Factor34 + 6*mu1**2*mu2 - 3*mu1**4)/(noiV**2)-3
/home/jason/.virtualenvs/ifd/lib/python3.8/site-packages/pyIFD/NOI2.py:217: RuntimeWarning: invalid value encountered in true_divide
  noiK = (Factor34 + 6*mu1**2*mu2 - 3*mu1**4)/(noiV**2)-3
/home/jason/.virtualenvs/ifd/lib/python3.8/site-packages/pyIFD/NOI2.py:222: RuntimeWarning: divide by zero encountered in true_divide
  b = np.mean(1/noiV,2)
/home/jason/.virtualenvs/ifd/lib/python3.8/site-packages/pyIFD/NOI2.py:223: RuntimeWarning: divide by zero encountered in true_divide
  c = np.mean(1/noiV**2,2)
/home/jason/.virtualenvs/ifd/lib/python3.8/site-packages/pyIFD/NOI2.py:224: RuntimeWarning: invalid value encountered in true_divide
  d = np.mean(np.sqrt(noiK)/noiV,2)
/home/jason/.virtualenvs/ifd/lib/python3.8/site-packages/pyIFD/NOI2.py:227: RuntimeWarning: overflow encountered in multiply
  sqrtK = (a*c - b*d)/(c-b*b)
/home/jason/.virtualenvs/ifd/lib/python3.8/site-packages/pyIFD/NOI2.py:227: RuntimeWarning: invalid value encountered in subtract
  sqrtK = (a*c - b*d)/(c-b*b)
/home/jason/.virtualenvs/ifd/lib/python3.8/site-packages/pyIFD/NOI2.py:217: RuntimeWarning: divide by zero encountered in true_divide
  noiK = (Factor34 + 6*mu1**2*mu2 - 3*mu1**4)/(noiV**2)-3
/home/jason/.virtualenvs/ifd/lib/python3.8/site-packages/pyIFD/NOI2.py:217: RuntimeWarning: invalid value encountered in true_divide
  noiK = (Factor34 + 6*mu1**2*mu2 - 3*mu1**4)/(noiV**2)-3
/home/jason/.virtualenvs/ifd/lib/python3.8/site-packages/pyIFD/NOI2.py:222: RuntimeWarning: divide by zero encountered in true_divide
  b = np.mean(1/noiV,2)
/home/jason/.virtualenvs/ifd/lib/python3.8/site-packages/pyIFD/NOI2.py:223: RuntimeWarning: divide by zero encountered in true_divide
  c = np.mean(1/noiV**2,2)
/home/jason/.virtualenvs/ifd/lib/python3.8/site-packages/pyIFD/NOI2.py:223: RuntimeWarning: overflow encountered in true_divide
  c = np.mean(1/noiV**2,2)
/home/jason/.virtualenvs/ifd/lib/python3.8/site-packages/pyIFD/NOI2.py:224: RuntimeWarning: invalid value encountered in true_divide
  d = np.mean(np.sqrt(noiK)/noiV,2)
/home/jason/.virtualenvs/ifd/lib/python3.8/site-packages/pyIFD/NOI2.py:227: RuntimeWarning: overflow encountered in multiply
  sqrtK = (a*c - b*d)/(c-b*b)
/home/jason/.virtualenvs/ifd/lib/python3.8/site-packages/pyIFD/NOI2.py:227: RuntimeWarning: invalid value encountered in multiply
  sqrtK = (a*c - b*d)/(c-b*b)
/home/jason/.virtualenvs/ifd/lib/python3.8/site-packages/pyIFD/NOI2.py:227: RuntimeWarning: invalid value encountered in subtract
  sqrtK = (a*c - b*d)/(c-b*b)
/home/jason/.virtualenvs/ifd/lib/python3.8/site-packages/pyIFD/NOI2.py:227: RuntimeWarning: invalid value encountered in true_divide
  sqrtK = (a*c - b*d)/(c-b*b)`


###NOI4###

Significant failures under 0.9 similarity. Some mismatched types in compare? 

###NOI5###

  File "/home/jason/.virtualenvs/ifd/lib/python3.8/site-packages/pyIFD/NOI5.py", line 310, in PCANoise
    (label64[i,j], Noise_64[i,j]) =  PCANoiseLevelEstimator(Ib,5)
  File "/home/jason/.virtualenvs/ifd/lib/python3.8/site-packages/pyIFD/NOI5.py", line 186, in PCANoiseLevelEstimator
    upper_bound = ComputeUpperBound( block_info )
  File "/home/jason/.virtualenvs/ifd/lib/python3.8/site-packages/pyIFD/NOI5.py", line 144, in ComputeUpperBound
    nozeroindex = np.min(np.max(np.where(block_info[:,0]== 0)[0])+1,np.shape(block_info)[0])
  File "<__array_function__ internals>", line 5, in amin
  File "/home/jason/.virtualenvs/ifd/lib/python3.8/site-packages/numpy/core/fromnumeric.py", line 2879, in amin
    return _wrapreduction(a, np.minimum, 'min', axis, None, out,
  File "/home/jason/.virtualenvs/ifd/lib/python3.8/site-packages/numpy/core/fromnumeric.py", line 84, in _wrapreduction
    return reduction(axis=axis, out=out, **passkwargs)
  File "/home/jason/.virtualenvs/ifd/lib/python3.8/site-packages/numpy/core/_methods.py", line 44, in _amin
    return umr_minimum(a, axis, None, out, keepdims, initial, where)
numpy.AxisError: axis 947 is out of bounds for array of dimension 0

