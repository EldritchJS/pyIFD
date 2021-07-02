import numpy as np
def svmdecision(Xnew,svm_struct):
    sv=svm_struct[0]
    alphaHat=svm_struct[1]
    bias=svm_struct[2][0][0]
    f=np.dot(np.tanh(sv @ np.transpose(Xnew)-1),alphaHat)+bias
    return f
    