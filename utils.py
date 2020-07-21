import random as r


def independent(x, S):
    import sympy as sp
    mat = []
    for i in range(len(S[0])):
        temp = []
        for j in range(len(S)):
            temp.append(S[j][i])
        temp.append(x[i])
        mat.append(temp)
    mat = sp.Matrix(mat)
    res = mat.columnspace()
    return len(res) == len(S) + 1


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
