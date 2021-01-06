from tensorflow import keras
from nnexpy import KnotDescriptor, RandomNetwork
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits import mplot3d
import pickle

mypath = "./models/"

if not os.path.exists(mypath + "random_networks"):
    os.makedirs(mypath + "random_networks")

trajList = [('trefoil', KnotDescriptor.fromTemplate('trefoil3D'), KnotDescriptor.fromTemplate('trefoil4D')), ('figureEight', KnotDescriptor.fromTemplate(
    'figureEight3D'), KnotDescriptor.fromTemplate('figureEight4D')), ('unknot', KnotDescriptor.fromTemplate('circular3D'), KnotDescriptor.fromTemplate('circular4D'))]

for traj in trajList:
    for k in range(3):
        if not os.path.exists(mypath + "random_networks/instance_" + str(k) + "/"):
            os.makedirs(mypath + "random_networks/instance_" + str(k) + "/")
        mypath_init = "./models/random_networks/instance_" + str(k) + "/"

        for trainingnumb in range(100):
            if not os.path.exists(mypath_init + "training_" + str(trainingnumb) + "/"):
                os.makedirs(mypath_init + "training_" +
                            str(trainingnumb) + "/")
            mypath = mypath_init + "training_" + str(trainingnumb) + "/"
            if (k < 2):
                trajectory = traj[1].trajectory
            else:
                trajectory = traj[2].trajectory

            for depth in [1, 2, 4, 8]:
                if (k < 2):
                    model = RandomNetwork(
                        depth=depth, width=8, inputDimension=3)
                else:
                    model = RandomNetwork(
                        depth=depth, width=8, inputDimension=4)

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
                        next = np.tanh(
                            np.dot(np.transpose(layersWeights[i]), init) + layersBias[i])
                        res[i].append(next)
                        init = next

                pcaRes = []
                for _ in range(len(layersWeights) - 1):
                    pcaRes.append([])

                for i in range(len(res)):
                    pca = PCA(n_components=3)
                    principalComponents = pca.fit_transform(res[i])
                    pcaRes[i] = principalComponents

                for i, x in enumerate(pcaRes):
                    desc = KnotDescriptor.fromTrajectory(x)
                    determinant = desc.determinant()

                    information = {'trajectory': traj[0],
                                   'layerNumber': len(pcaRes) - 1, 'layerIndex': i}

                    with open(mypath + traj[0] +
                              str(len(pcaRes) - 1) + "_" + str(i) + '_layer.pkl', 'wb') as output:
                        pickle.dump(information, output,
                                    pickle.HIGHEST_PROTOCOL)
                        pickle.dump(x, output,
                                    pickle.HIGHEST_PROTOCOL)
                        pickle.dump(desc.component, output,
                                    pickle.HIGHEST_PROTOCOL)
                        pickle.dump(desc.crossing, output,
                                    pickle.HIGHEST_PROTOCOL)
                        pickle.dump(determinant, output,
                                    pickle.HIGHEST_PROTOCOL)
