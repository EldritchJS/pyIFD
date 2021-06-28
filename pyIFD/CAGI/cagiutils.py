def mat2gray(A):
    A-=A.min()
    if(A.max()==0):
        return A
    return A/A.max()

