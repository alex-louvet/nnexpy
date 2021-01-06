from tensorflow import keras
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits import mplot3d
import pickle
from nnexpy import DataDescriptor, DataInstance, DataPoint, KnotDescriptor

for k in range(3):
    mypath_init = "./models/instance_" + str(k) + "/"

    with open(mypath_init + 'data_descriptor.pkl', 'rb') as input:
        centerList = pickle.load(input)
        radiusList = pickle.load(input)
        bounds = pickle.load(input)
        randomSeed = pickle.load(input)

    dataDescriptor = DataDescriptor(nHoles=len(centerList), centerList=centerList,
                                    radiusList=radiusList, bounds=bounds)

    instance = dataDescriptor.generateData(classNumber=2, nPoints=50000)
    data_betti = instance.bettiNumbers()
    test = dataDescriptor.generateTestData(nPoints=50000)
    print(data_betti)

    for trainingnumb in range(15):
        mypath = mypath_init + "training_" + str(trainingnumb) + "/"
        if not os.path.exists(mypath + "figure"):
            os.makedirs(mypath + "figure")
        if not os.path.exists(mypath + "pca"):
            os.makedirs(mypath + "pca")

        for file in os.listdir(mypath):
            if file.endswith(".h5") and not file.startswith('16'):
                modelPath = os.path.join(mypath, file)

                if (k < 2):
                    trajectory = KnotDescriptor.fromTemplate(
                        'figureEight3D', nPoints=5000)
                else:
                    trajectory = KnotDescriptor.fromTemplate(
                        'figureEight4D', nPoints=5000)

                model = keras.models.load_model(modelPath)
                print(modelPath)
                predictedTest = test.predict(model, verbose=1)
                modelBetti = predictedTest.bettiNumbers()
                print(modelBetti)

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
                    print(determinant)
                    """
                    information = {'trajectory': 'figureEight',
                                   'layerNumber': len(pcaRes) - 1, 'layerIndex': i}

                    with open(mypath + 'pca/' + 'figureEight' +
                              str(len(pcaRes) - 1) + "_" + str(i) + '_layer.pkl', 'wb') as output:
                        pickle.dump(information, output,
                                    pickle.HIGHEST_PROTOCOL)
                        pickle.dump(x, output,
                                    pickle.HIGHEST_PROTOCOL)
                        pickle.dump(desc.component, output,
                                    pickle.HIGHEST_PROTOCOL)
                        pickle.dump(desc.crossing, output,
                                    pickle.HIGHEST_PROTOCOL)
                        pickle.dump(desc.determinant, output,
                                    pickle.HIGHEST_PROTOCOL)
                        pickle.dump(data_betti, output,
                                    pickle.HIGHEST_PROTOCOL)
                        pickle.dump(modelBetti, output,
                                    pickle.HIGHEST_PROTOCOL)
                    fig = plt.figure()
                    ax = plt.axes(projection="3d")
                    ax.scatter3D([e[0] for e in x], [e[1]
                                                     for e in x], [e[2] for e in x], c=range(len(x)), cmap="hsv", s=4)

                    fig.savefig(mypath + "figure/" + 'figureEight' +
                                str(len(pcaRes) - 1) + "_" + str(i) + '_layer')
                    fig.clf()
                    plt.close(fig)
                    """
