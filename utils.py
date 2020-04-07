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
