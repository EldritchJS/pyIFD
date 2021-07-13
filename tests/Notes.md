###ADQ1###

  File "/home/jason/.virtualenvs/ifd/lib/python3.8/site-packages/pyIFD/ADQ1.py", line 152, in detectDQ_JPEG
    FFTPeak=np.argmax(FFT)+1
  File "<__array_function__ internals>", line 5, in argmax
  File "/home/jason/.virtualenvs/ifd/lib/python3.8/site-packages/numpy/core/fromnumeric.py", line 1195, in argmax
    return _wrapfunc(a, 'argmax', axis=axis, out=out)
  File "/home/jason/.virtualenvs/ifd/lib/python3.8/site-packages/numpy/core/fromnumeric.py", line 57, in _wrapfunc
    return bound(*args, **kwds)
ValueError: attempt to get argmax of an empty sequence



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

```  File "validate_algo.py", line 97, in main   
 similarity.append(comp(matDispImages[i],pyDispImages[i]))
  File "/home/jason/.virtualenvs/ifd/lib/python3.8/site-packages/skimage/metrics/_structural_similarity.py", line 89, in structural_similarity
    check_shape_equality(im1, im2)
  File "/home/jason/.virtualenvs/ifd/lib/python3.8/site-packages/skimage/_shared/utils.py", line 294, in check_shape_equality
    raise ValueError('Input images must have the same dimensions.')
ValueError: Input images must have the same dimensions.```

