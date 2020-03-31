import matplotlib.pyplot as plt


def circular_trajectory(*args, **kwargs):
    import numpy as np
    nPoints = kwargs.get('nPoints', 1000)
    radius = kwargs.get('radius', 1)
    center = kwargs.get('center', (0, 0))
    res = np.linspace(0, 1, nPoints)
    return np.array([(radius * np.cos(np.pi * 2 * x) + center[0], radius * np.sin(np.pi * 2 * x) + center[1]) for x in res])


def tanh(V):
    import numpy as np
    return (np.exp(V) - np.exp(-1 * V)) / (np.exp(V) + np.exp(-1 * V))
