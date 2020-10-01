import random as r


def independent(x, S):
    import numpy as np
    mat = S.copy()
    mat.append(x)
    rank = np.linalg.matrix_rank(mat)
    return rank == len(mat[0])


def findPointStructDimension(pointList):
    import numpy as np
    A = [list(x) for x in pointList]
    B = [A[0]]
    A = A[1:]
    while len(A) > 0 and len(B) < len(B[0]):
        head = A[0]
        A = A[1:]
        if independent(head, B):
            B.append(head)
    return len(B) - 1
