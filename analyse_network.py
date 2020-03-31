from tensorflow import keras
from utils import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os

mypath = "./models/test/"
if not os.path.exists(mypath + "figure"):
    os.makedirs(mypath + "figure")

for file in os.listdir(mypath):
    if file.endswith(".h5"):
        modelPath = os.path.join(mypath, file)

        trajectory = circular_trajectory(nPoints=50000)

        model = keras.models.load_model(modelPath)

        print(model.summary())

        layersWeights = []
        layersBias = []

        for layer in model.layers:
            temp = layer.get_weights()
            layersWeights.append(temp[0])
            layersBias.append(temp[1])

        res = []
        for _ in range(len(layersWeights) - 1):
            res.append([])

        for x in trajectory:
            init = x
            for i in range(len(layersWeights) - 1):
                next = tanh(
                    np.dot(np.transpose(layersWeights[i]), init) + layersBias[i])
                res[i].append(next)
                init = next

        pcaRes = []
        for _ in range(len(layersWeights) - 1):
            pcaRes.append([])

        for i in range(len(res)):
            pca = PCA(n_components=len(layersWeights[0]))
            principalComponents = pca.fit_transform(res[i])
            pcaRes[i] = principalComponents

        for i, x in enumerate(pcaRes):
            plt.scatter([e[0] for e in x], [e[1]
                                            for e in x], c=range(len(x)), cmap="hsv", s=4)

            plt.savefig(mypath + "figure/" + 'pca_' +
                        str(len(pcaRes) - 1) + "_" + str(i) + '_layer')
            plt.clf()
