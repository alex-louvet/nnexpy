import pickle
from nnexpy import DataDescriptor, DataInstance
from tensorflow import keras
from os import walk, path
import time as t
import sys
import gc
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

score = [0, 0, 0, 0, 0, 0, 0, 0]

instanceNumber = int(sys.argv[1])

mypath = './models/instance_' + \
    str(instanceNumber) + '/'

with open(mypath + 'data_descriptor.pkl', 'rb') as input:
    centerList = pickle.load(input)
    radiusList = pickle.load(input)
    bounds = pickle.load(input)
    holeDimension = pickle.load(input)
    orientation = pickle.load(input)
    randomSeed = pickle.load(input)

dataDescriptor = DataDescriptor(nHoles=len(centerList), centerList=centerList, radiusList=radiusList,
                                bounds=bounds, holeDimension=holeDimension, random=randomSeed, orientation=orientation)

THRESHOLD = 0.04

for i in range(len(radiusList)):
    for j in range(len(radiusList[i])-1):
        THRESHOLD = min(
            THRESHOLD, radiusList[i][j + 1][0] - radiusList[i][j][1])

for i in range(len(centerList)):
    for j in range(i+1, len(centerList)):
        THRESHOLD = min(THRESHOLD, centerList[i].distanceTo(
            centerList[j]) - (radiusList[i][-1][1] + radiusList[j][-1][1]))

THRESHOLD = 0.9*THRESHOLD
print(THRESHOLD)

instance = dataDescriptor.generateData(
    classNumber=2, nPoints=50000, random=randomSeed)
data_betti = instance.newBettiNumbers(
    threshold=THRESHOLD, nPoints=min(10000, 2000 * 0.04 / THRESHOLD))
test = dataDescriptor.generateData(
    nPoints=50000, random=randomSeed)

for directory in [x[0] for x in walk(mypath)][1:]:
    model1 = keras.models.load_model(directory + '/1layer.h5')
    model2 = keras.models.load_model(directory + '/2layer.h5')
    model3 = keras.models.load_model(directory + '/3layer.h5')
    model4 = keras.models.load_model(directory + '/4layer.h5')
    model5 = keras.models.load_model(directory + '/5layer.h5')
    model6 = keras.models.load_model(directory + '/6layer.h5')
    model7 = keras.models.load_model(directory + '/7layer.h5')
    model8 = keras.models.load_model(directory + '/8layer.h5')

    predictedTest = test.predict(model1, verbose=1)
    temp1 = predictedTest.newBettiNumbers(
        threshold=THRESHOLD, nPoints=min(10000, 2000 * 0.04 / THRESHOLD))
    if temp1 == data_betti:
        score[0] += 1

    predictedTest = test.predict(model2, verbose=1)
    temp2 = predictedTest.newBettiNumbers(
        threshold=THRESHOLD, nPoints=min(10000, 2000 * 0.04 / THRESHOLD))
    if temp2 == data_betti:
        score[1] += 1

    predictedTest = test.predict(model1, verbose=1)
    temp3 = predictedTest.newBettiNumbers(
        threshold=THRESHOLD, nPoints=min(10000, 2000 * 0.04 / THRESHOLD))
    if temp3 == data_betti:
        score[2] += 1

    predictedTest = test.predict(model4, verbose=1)
    temp4 = predictedTest.newBettiNumbers(
        threshold=THRESHOLD, nPoints=min(10000, 2000 * 0.04 / THRESHOLD))
    if temp4 == data_betti:
        score[3] += 1

    predictedTest = test.predict(model1, verbose=1)
    temp5 = predictedTest.newBettiNumbers(
        threshold=THRESHOLD, nPoints=min(10000, 2000 * 0.04 / THRESHOLD))
    if temp5 == data_betti:
        score[4] += 1

    predictedTest = test.predict(model6, verbose=1)
    temp6 = predictedTest.newBettiNumbers(
        threshold=THRESHOLD, nPoints=min(10000, 2000 * 0.04 / THRESHOLD))
    if temp6 == data_betti:
        score[5] += 1

    predictedTest = test.predict(model1, verbose=1)
    temp7 = predictedTest.newBettiNumbers(
        threshold=THRESHOLD, nPoints=min(10000, 2000 * 0.04 / THRESHOLD))
    if temp7 == data_betti:
        score[6] += 1

    predictedTest = test.predict(model8, verbose=1)
    temp8 = predictedTest.newBettiNumbers(
        threshold=THRESHOLD, nPoints=min(10000, 2000 * 0.04 / THRESHOLD))
    if temp8 == data_betti:
        score[7] += 1

    file = open(directory + '/betti.txt', "w")
    file.write(str(temp1) + '\n')
    file.write(str(temp2) + '\n')
    file.write(str(temp3) + '\n')
    file.write(str(temp4) + '\n')
    file.write(str(temp5) + '\n')
    file.write(str(temp6) + '\n')
    file.write(str(temp7) + '\n')
    file.write(str(temp8) + '\n')
    file.close()

file = open(mypath + 'information.txt', "a")
file.write('data betti numbers: ' + str(data_betti) + "\n")
file.write('# of correspondence: ' + str(score) + "\n")
file.close()

del predictedTest
gc.collect()
tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()
