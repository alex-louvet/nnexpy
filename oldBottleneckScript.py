import pickle
from generate_data_dimension import *
from tensorflow import keras
from os import walk, path
import time as t
import sys
import gc
import tensorflow as tf
import gudhi
import numpy as np
tf.compat.v1.disable_eager_execution()

orientation = [[[], [0]], None, None, [[0]], [[], [2]], None, None,
               [[1]], [[], [3]], None, None, [[1]]]

holeDimList = [[2, 1], [2, 2], [2], [1], [3, 2],
               [3, 3], [3], [2], [4, 3], [4, 4], [4], [3]]

score = [0, 0, 0, 0, 0, 0, 0, 0, 0]

instanceNumber = int(sys.argv[1])
mypath = '/home/alex/nn-expressiveness/models/instance_' + \
    str(instanceNumber) + '/'

with open(mypath + 'data_descriptor.pkl', 'rb') as input:
    centerList = pickle.load(input)
    radiusList = pickle.load(input)
    bounds = pickle.load(input)
    randomSeed = pickle.load(input)

dataDescriptor = DataDescriptor(nHoles=len(centerList), centerList=centerList,
                                radiusList=radiusList, random=t.time(), bounds=bounds, holeDimension=holeDimList[instanceNumber])

instance = dataDescriptor.generateData(
    classNumber=2, pointsNumber=50000, orientation=orientation[instanceNumber])
test = dataDescriptor.generateData(
    pointsNumber=50000, orientation=orientation[instanceNumber])

k = [0, 0, 0, 0, 0, 0, 0, 0, 0]

for directory in [x[0] for x in walk(mypath)][1:]:
    print(directory)
    model1 = keras.models.load_model(directory + '/1layer.h5')
    model2 = keras.models.load_model(directory + '/2layer.h5')
    model4 = keras.models.load_model(directory + '/4layer.h5')
    model6 = keras.models.load_model(directory + '/6layer.h5')
    model8 = keras.models.load_model(directory + '/8layer.h5')
    model10 = keras.models.load_model(directory + '/10layer.h5')
    model12 = keras.models.load_model(directory + '/12layer.h5')
    model14 = keras.models.load_model(directory + '/14layer.h5')
    model16 = keras.models.load_model(directory + '/16layer.h5')

    predictedTest = test.predict(model1, verbose=1)
    bottle1 = predictedTest.computeBottleNeckDistance(instance, nPoints=10000)
    if bottle1 != -1:
        k[0] += 1
        score[0] += bottle1

    predictedTest = test.predict(model2, verbose=1)
    bottle2 = predictedTest.computeBottleNeckDistance(instance, nPoints=10000)
    if bottle2 != -1:
        k[1] += 1
        score[1] += bottle2

    predictedTest = test.predict(model4, verbose=1)
    bottle4 = predictedTest.computeBottleNeckDistance(instance, nPoints=10000)
    if bottle4 != -1:
        k[2] += 1
        score[2] += bottle4

    predictedTest = test.predict(model6, verbose=1)
    bottle6 = predictedTest.computeBottleNeckDistance(instance, nPoints=10000)
    if bottle6 != -1:
        k[3] += 1
        score[3] += bottle6

    predictedTest = test.predict(model8, verbose=1)
    bottle8 = predictedTest.computeBottleNeckDistance(instance, nPoints=10000)
    if bottle8 != -1:
        k[4] += 1
        score[4] += bottle8

    predictedTest = test.predict(model10, verbose=1)
    bottle10 = predictedTest.computeBottleNeckDistance(instance, nPoints=10000)
    if bottle10 != -1:
        k[5] += 1
        score[5] += bottle10

    predictedTest = test.predict(model12, verbose=1)
    bottle12 = predictedTest.computeBottleNeckDistance(instance, nPoints=10000)
    if bottle12 != -1:
        k[6] += 1
        score[6] += bottle12

    predictedTest = test.predict(model14, verbose=1)
    bottle14 = predictedTest.computeBottleNeckDistance(instance, nPoints=10000)
    if bottle14 != -1:
        k[7] += 1
        score[7] += bottle14

    predictedTest = test.predict(model16, verbose=1)
    bottle16 = predictedTest.computeBottleNeckDistance(instance, nPoints=10000)
    if bottle16 != -1:
        k[8] += 1
        score[8] += bottle16

    print(score)
    print(k)

    file = open(directory + '/bottleneck.txt', "w")
    file.write(str(bottle1) + '\n')
    file.write(str(bottle2) + '\n')
    file.write(str(bottle4) + '\n')
    file.write(str(bottle6) + '\n')
    file.write(str(bottle8) + '\n')
    file.write(str(bottle10) + '\n')
    file.write(str(bottle12) + '\n')
    file.write(str(bottle14) + '\n')
    file.write(str(bottle16) + '\n')
    file.close()

score = [score[i]/k[i] for i in range(len(score))]
file = open(mypath + 'information.txt', "a")
file.write('bottleneck avg: ' + str(score) + "\n")
file.close()

del predictedTest
gc.collect()
tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()
