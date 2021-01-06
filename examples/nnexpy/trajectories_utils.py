def circular_trajectory2D(*args, **kwargs):
    import numpy as np
    nPoints = kwargs.get('nPoints', 1000)
    radius = kwargs.get('radius', 1)
    center = kwargs.get('center', (0, 0))
    res = np.linspace(0, 1, nPoints)
    return np.array([(radius * np.cos(np.pi * 2 * x) + center[0], radius * np.sin(np.pi * 2 * x) + center[1]) for x in res])


def tanh(V):
    import numpy as np
    return (np.exp(V) - np.exp(-1 * V)) / (np.exp(V) + np.exp(-1 * V))


def gaussian(V, *args, **kwargs):
    import numpy as np
    a = kwargs.get('a', 1)
    b = kwargs.get('b', 0)
    c = kwargs.get('c', 1)
    return (a * np.exp(-1 * (V - b)**2 / (2 * c**2)))


def circular3D(*args, **kwargs):
    import numpy as np
    import random as r
    nPoints = kwargs.get('nPoints', 1000)
    radius = kwargs.get('radius', 1)
    center = kwargs.get('center', (0, 0, 0))
    temp = kwargs.get('height', r.random())
    res = np.linspace(0, 1, nPoints)
    height = np.linspace(-10, 10, nPoints)
    return np.array([(radius * np.cos(np.pi * 2 * x) + center[0], radius * np.sin(np.pi * 2 * x) + center[1], gaussian(height[i], a=temp, c=3) + center[2]) for i, x in enumerate(res)])


def circular4D(*args, **kwargs):
    import numpy as np
    import random as r
    nPoints = kwargs.get('nPoints', 1000)
    radius = kwargs.get('radius', 1)
    center = kwargs.get('center', (0, 0, 0, 0))
    temp1 = kwargs.get('height1', r.random())
    temp2 = kwargs.get('height2', r.random())
    res = np.linspace(0, 1, nPoints)
    height = np.linspace(-10, 10, nPoints)
    return np.array([(radius * np.cos(np.pi * 2 * x) + center[0], radius * np.sin(np.pi * 2 * x) + center[1], gaussian(height[i], a=temp1, c=3) + center[2], gaussian(height[i], a=temp2, c=3) + center[3]) for i, x in enumerate(res)])


# https://github.com/kawache/Python-B-spline-examples
def bspline(ctr, *args, **kwargs):
    import numpy as np
    from scipy import interpolate
    # x = np.arange(0, 2*np.pi+np.pi/4, 2*np.pi/8)
    # y = np.sin(x)
    closed = kwargs.get('closed', True)
    nPoints = kwargs.get('nPoints', 1000)

    dimension = len(ctr[0])
    coordinatesList = []
    for _ in range(dimension):
        coordinatesList.append([])

    ctr = np.array(ctr)

    for x in ctr:
        if len(x) != dimension:
            raise ValueError(
                'points must live in the same dimension')
        else:
            for i in range(len(x)):
                coordinatesList[i].append(x[i])

    if closed:
        for i, x in enumerate(coordinatesList):
            coordinatesList[i] = np.append(x, x[0])

    tck, u = interpolate.splprep(coordinatesList, k=3, s=0)
    u = np.linspace(0, 1, num=nPoints, endpoint=True)
    return interpolate.splev(u, tck)


def trefoil3D(*args, **kwargs):
    import numpy as np
    nPoints = kwargs.get('nPoints', 1000)
    radius = kwargs.get('radius', 1)
    center = kwargs.get('center', (0, 0, 0))
    trefoil = np.array([(0, 1, 0.5), (-0.33, 0.33, 1), (0, -0.5, 0), (1, -0.5, 0.5), (0.33,
                                                                                      0.33, 1), (-0.33, 0.33, 0), (-1, -0.5, 0.5), (0, -0.5, 1), (0.33, 0.33, 0)])
    trefoil = radius * trefoil + np.array(center)
    temp = bspline(trefoil, nPoints=nPoints)
    return np.array([(temp[0][i], temp[1][i], temp[2][i]) for i in range(len(temp[0]))])


def trefoil4D(*args, **kwargs):
    import numpy as np
    import random as r
    nPoints = kwargs.get('nPoints', 1000)
    radius = kwargs.get('radius', 1)
    center = kwargs.get('center', (0, 0, 0, 0))
    random1 = r.random() - 0.5
    random2 = r.random() - 0.5
    fourthd = np.array(list(np.linspace(min(random1, random2), max(random1, random2), num=5)) +
                       list(np.linspace(max(random1, random2), min(random1, random2), num=4)))
    trefoil = np.array([(0, 1, 0.5, fourthd[0]), (-0.33, 0.33, 1, fourthd[1]), (0, -0.5, 0, fourthd[2]), (1, -0.5, 0.5, fourthd[3]), (0.33,
                                                                                                                                      0.33, 1, fourthd[4]), (-0.33, 0.33, 0, fourthd[5]), (-1, -0.5, 0.5, fourthd[6]), (0, -0.5, 1, fourthd[7]), (0.33, 0.33, 0, fourthd[8])])
    trefoil = radius * trefoil + np.array(center)
    temp = bspline(trefoil, nPoints=nPoints)
    return np.array([(temp[0][i], temp[1][i], temp[2][i], temp[3][i]) for i in range(len(temp[0]))])


def figureEight3D(*args, **kwargs):
    import numpy as np
    nPoints = kwargs.get('nPoints', 1000)
    radius = kwargs.get('radius', 1)
    center = kwargs.get('center', (0, 0, 0))
    figureEight = np.array([(0, 1, 0.5), (-0.33, 0.66, 0), (0, 0, 1), (0.25, -0.5, 0.5), (0, -0.66, 0), (-1, 0, 0.5),
                            (-0.33, 0.66, 1), (0.33, 0.66, 0), (1, 0, 0.5), (0, -0.66, 1), (-0.25, -0.5, 0.5), (0, 0, 0), (0.33, 0.66, 1)])
    figureEight = radius * figureEight + np.array(center)
    temp = bspline(figureEight, nPoints=nPoints)
    return np.array([(temp[0][i], temp[1][i], temp[2][i]) for i in range(len(temp[0]))])


def figureEight4D(*args, **kwargs):
    import numpy as np
    import random as r
    nPoints = kwargs.get('nPoints', 1000)
    radius = kwargs.get('radius', 1)
    center = kwargs.get('center', (0, 0, 0, 0))
    random1 = r.random() - 0.5
    random2 = r.random() - 0.5
    fourthd = np.array(list(np.linspace(min(random1, random2), max(random1, random2), num=8)) +
                       list(np.linspace(max(random1, random2), min(random1, random2), num=7)))
    figureEight = np.array([(0, 1, 0.5, fourthd[0]), (-0.33, 0.66, 0, fourthd[1]), (0, 0, 1, fourthd[2]), (0.25, -0.5, 0.5, fourthd[3]), (0, -0.66, 0, fourthd[4]), (-1, 0, 0.5, fourthd[5]), (-0.33,
                                                                                                                                                                                               0.66, 1, fourthd[6]), (0.33, 0.66, 0, fourthd[7]), (1, 0, 0.5, fourthd[8]), (0, -0.66, 1, fourthd[9]), (-0.25, -0.5, 0.5, fourthd[10]), (0, 0, 0, fourthd[11]), (0.33, 0.66, 1, fourthd[12])])
    figureEight = radius * figureEight + np.array(center)
    temp = bspline(figureEight, nPoints=nPoints)
    return np.array([(temp[0][i], temp[1][i], temp[2][i], temp[3][i]) for i in range(len(temp[0]))])


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def vectorProduct(A, B, C, D):
    a = (B[0] - A[0], B[1] - A[1], B[2] - A[2])
    b = (D[0] - C[0], D[1] - C[1], D[2] - C[2])
    s = (a[1] * b[2] - a[2] * b[1], a[2] * b[0] -
         a[0] * b[2], a[0] * b[1] - a[1] * b[0])
    return (int(A[2] == max(A[2], C[2], D[2]) or B[2] == max(B[2], C[2], D[2])) * 2 - 1) * s[2]


# https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def ABIsUp(A, B, C, D):
    normX = line_intersection((A, B), (C, D))[0]/(B[0] - A[0])
    return A[2]*normX + (1-normX)*B[2] >= C[2]*normX + (1-normX)*D[2]
