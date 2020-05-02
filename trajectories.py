import matplotlib.pyplot as plt


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
    return bspline(trefoil, nPoints=nPoints)


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
    return bspline(trefoil, nPoints=nPoints)


def figureEight3D(*args, **kwargs):
    import numpy as np
    nPoints = kwargs.get('nPoints', 1000)
    radius = kwargs.get('radius', 1)
    center = kwargs.get('center', (0, 0, 0))
    trefoil = np.array([(0, 1, 0.5), (-0.33, 0.66, 0), (0, 0, 1), (0.25, -0.5, 0.5), (0, -0.66, 0), (-1, 0, 0.5),
                        (-0.33, 0.66, 1), (0.33, 0.66, 0), (1, 0, 0.5), (0, -0.66, 1), (-0.25, -0.5, 0.5), (0, 0, 0), (0.33, 0.66, 1)])
    trefoil = radius * trefoil + np.array(center)
    return bspline(trefoil, nPoints=nPoints)


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
    trefoil = np.array([(0, 1, 0.5, fourthd[0]), (-0.33, 0.66, 0, fourthd[1]), (0, 0, 1, fourthd[2]), (0.25, -0.5, 0.5, fourthd[3]), (0, -0.66, 0, fourthd[4]), (-1, 0, 0.5, fourthd[5]), (-0.33,
                                                                                                                                                                                           0.66, 1, fourthd[6]), (0.33, 0.66, 0, fourthd[7]), (1, 0, 0.5, fourthd[8]), (0, -0.66, 1, fourthd[9]), (-0.25, -0.5, 0.5, fourthd[10]), (0, 0, 0, fourthd[11]), (0.33, 0.66, 1, fourthd[12])])
    trefoil = radius * trefoil + np.array(center)
    return bspline(trefoil, nPoints=nPoints)


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


def ABIsUp(A, B, C, D):
    return A[2] == max(A[2], C[2], D[2]) or B[2] == max(B[2], C[2], D[2])


class knotDescriptor(object):
    def __init__(self, trajectory):
        crossing = []
        component = []
        already_seen = []
        for i in range(len(trajectory) - 1):
            for j in range(len(trajectory) - 1):
                if intersect(trajectory[i + 1], trajectory[i], trajectory[j + 1], trajectory[j]) and abs(i - j) > 2 and not (i == 0 and j == len(trajectory) - 2 or i == len(trajectory) - 2 and j == 0):
                    if vectorProduct(trajectory[i + 1], trajectory[i], trajectory[j + 1], trajectory[j]) > 0 and not (i, j) in already_seen and not (j, i) in already_seen:
                        crossing.append(('r', i, j))
                        already_seen.append((i, j))
                    elif vectorProduct(trajectory[i + 1], trajectory[i], trajectory[j + 1], trajectory[j]) < 0 and not (i, j) in already_seen and not (j, i) in already_seen:
                        crossing.append(('l', i, j))
                        already_seen.append((i, j))
                    if not ABIsUp(trajectory[i + 1], trajectory[i], trajectory[j + 1], trajectory[j]):
                        if len(component) == 0:
                            component.append((0, i))
                        else:
                            component.append((component[-1][1] + 1, i))
        if (len(component) > 1):
            component[0] = (component[-1][1] + 1, component[1][0] - 1)
        else:
            component = [(0, len(trajectory) - 1)]

        self.crossing = crossing
        self.component = component

    def findCrossingComponents(self, crossing):
        if not crossing in self.crossing:
            raise valueError(
                "crossing must be one of the crossings of the knot")
        c1 = 0
        c2 = 0
        c3 = 0
        for i, x in enumerate(self.component):
            if (crossing[1] == x[1] or crossing[2] == x[1]):
                c2 = i
                if i == len(self.component) - 1:
                    c3 = 0
                else:
                    c3 = i + 1
            if crossing[1] >= x[0] and crossing[1] < x[1] or crossing[2] >= x[0] and crossing[2] < x[1]:
                c1 = i

        return c1, c2, c3

    def determinant(self):
        import numpy as np
        import math as m
        if (len(self.crossing) == 0 or len(self.component) == 1):
            return 1
        matrix = np.zeros(
            (len(self.crossing), len(self.component)), dtype=int)
        for i, x in enumerate(self.crossing):
            c1, c2, c3 = knotDescriptor.findCrossingComponents(self, x)
            matrix[i][c1] = 2
            matrix[i][c2] = -1
            matrix[i][c3] = -1
        matrix = np.delete(matrix, 0, 0)
        matrix = np.delete(matrix, 0, 1)

        det = int(np.round(abs(np.linalg.det(matrix))))
        self.det = det
        return det
