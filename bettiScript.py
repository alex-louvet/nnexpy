import pickle
from generate_data_dimension import *
from tensorflow import keras
from os import walk, path
import time as t
import sys
import gc
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

holeDimList = [[2, 1], [2, 2], [2], [1], [3, 2],
               [3, 3], [3], [2], [4, 3], [4, 4], [4], [3]]

score = [0, 0, 0, 0, 0, 0, 0, 0, 0]

instanceNumber = int(sys.argv[1])

mypath = '/home/alex/nn-expressiveness/models/instance_' + str(instanceNumber) + '/'

with open(mypath + 'data_descriptor.pkl', 'rb') as input:
    centerList = pickle.load(input)
    radiusList = pickle.load(input)
    bounds = pickle.load(input)
    randomSeed = pickle.load(input)

dataDescriptor = DataDescriptor(nHoles=len(centerList), centerList=centerList,
                                radiusList=radiusList, random=t.time(), bounds=bounds, holeDimension=holeDimList[instanceNumber])

instance = dataDescriptor.generateData(classNumber=2, pointsNumber=50000)
data_betti = instance.newBettiNumbers(threshold=0.04, nPoints=5000)
print(data_betti)
test = dataDescriptor.generateTestData(pointsNumber=50000)

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
    temp1 = predictedTest.newBettiNumbers(threshold=0.04, nPoints=50000)
    if temp1 == data_betti:
        score[0] += 1

    predictedTest = test.predict(model2, verbose=1)
    temp2 = predictedTest.newBettiNumbers(threshold=0.04, nPoints=50000)
    if temp2 == data_betti:
        score[1] += 1

    predictedTest = test.predict(model4, verbose=1)
    temp4 = predictedTest.newBettiNumbers(threshold=0.04, nPoints=50000)
    if temp4 == data_betti:
        score[2] += 1

    predictedTest = test.predict(model6, verbose=1)
    temp6 = predictedTest.newBettiNumbers(threshold=0.04, nPoints=50000)
    if temp6 == data_betti:
        score[3] += 1

    predictedTest = test.predict(model8, verbose=1)
    temp8 = predictedTest.newBettiNumbers(threshold=0.04, nPoints=50000)
    if temp8 == data_betti:
        score[4] += 1

    predictedTest = test.predict(model10, verbose=1)
    temp10 = predictedTest.newBettiNumbers(threshold=0.04, nPoints=50000)
    if temp10 == data_betti:
        score[5] += 1

    predictedTest = test.predict(model12, verbose=1)
    temp12 = predictedTest.newBettiNumbers(threshold=0.04, nPoints=50000)
    if temp12 == data_betti:
        score[6] += 1

    predictedTest = test.predict(model14, verbose=1)
    temp14 = predictedTest.newBettiNumbers(threshold=0.04, nPoints=50000)
    if temp14 == data_betti:
        score[7] += 1

    predictedTest = test.predict(model16, verbose=1)
    temp16 = predictedTest.newBettiNumbers(threshold=0.04, nPoints=50000)
    if temp16 == data_betti:
        score[8] += 1

    file = open(directory + '/betti.txt', "w")
    file.write(str(temp1) + '\n')
    file.write(str(temp2) + '\n')
    file.write(str(temp4) + '\n')
    file.write(str(temp6) + '\n')
    file.write(str(temp8) + '\n')
    file.write(str(temp10) + '\n')
    file.write(str(temp12) + '\n')
    file.write(str(temp14) + '\n')
    file.write(str(temp16) + '\n')
    file.close()

file = open(mypath + 'information.txt', "a")
file.write('Data Betti Numbers: ' + str(data_betti) + "\n")
file.write('# of correspondence: ' + str(score) + "\n")
file.close()

del predictedTest
gc.collect()
tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()
