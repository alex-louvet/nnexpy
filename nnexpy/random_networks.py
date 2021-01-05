class RandomNetwork:
    def __init__(self, *args, **kwargs):
        import numpy as np
        inputDimension = kwargs.get('inputDimension', 2)
        depth = kwargs.get('depth', 1)
        outputDimension = kwargs.get('outputDimension', 1)
        width = kwargs.get('width', 8)
        self.layers = []
        bias = np.random.normal(0, 4/3, (width,))
        weights = np.random.normal(0, 4/3, (inputDimension, width))
        self.layers.append(RandomNetworkLayer([weights, bias]))
        for _ in range(depth):
            bias = np.random.normal(0, 4/3, (width,))
            weights = np.random.normal(0, 4/3, (width, width))
            self.layers.append(RandomNetworkLayer([weights, bias]))
        bias = np.random.normal(0, 4/3, (outputDimension,))
        weights = np.random.normal(0, 4/3, (width, outputDimension))
        self.layers.append(RandomNetworkLayer([weights, bias]))
        self.layers = np.array(self.layers)


class RandomNetworkLayer:
    def __init__(self, weights):
        self.weights = weights

    def get_weights(self):
        return self.weights
