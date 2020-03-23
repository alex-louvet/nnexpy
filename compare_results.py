import pickle
from generate_data import *
from tensorflow import keras
from os import walk, path
import time as t

score = [0, 0, 0, 0, 0]

mypath = './models/20_5000_beginning_training/'

with open(mypath + 'data_descriptor.pkl', 'rb') as input:
    centerList = pickle.load(input)
    radiusList = pickle.load(input)
    bounds = pickle.load(input)
    randomSeed = pickle.load(input)

dataDescriptor = DataDescriptor(nHoles=len(centerList), centerList=centerList,
                                radiusList=radiusList, random=t.time(), bounds=bounds)

instance = dataDescriptor.generateData(classNumber=2, pointsNumber=50000)
data_betti = instance.bettiNumbers(nPoints=2500)
print(data_betti)
test = dataDescriptor.generateTestData(pointsNumber=50000)

for directory in [x[0] for x in walk(mypath)][1:]:
    print(directory)
    model1 = keras.models.load_model(directory + '/1layer.h5')
    model2 = keras.models.load_model(directory + '/2layers.h5')
    model4 = keras.models.load_model(directory + '/4layers.h5')
    model8 = keras.models.load_model(directory + '/8layers.h5')
    model16 = keras.models.load_model(directory + '/16layers.h5')

    predictedTest = test.predict(model1, verbose=1)
    if predictedTest.bettiNumbers(nPoints=10000) == data_betti:
        score[0] += 1

    predictedTest = test.predict(model2, verbose=1)
    if predictedTest.bettiNumbers(nPoints=10000) == data_betti:
        score[1] += 1

    predictedTest = test.predict(model4, verbose=1)
    if predictedTest.bettiNumbers(nPoints=10000) == data_betti:
        score[2] += 1

    predictedTest = test.predict(model8, verbose=1)
    if predictedTest.bettiNumbers(nPoints=10000) == data_betti:
        score[3] += 1

    predictedTest = test.predict(model16, verbose=1)
    if predictedTest.bettiNumbers(nPoints=10000) == data_betti:
        score[4] += 1
    print(score)

file = open(mypath + 'information.txt', "a")
file.write('Data Betti Numbers: ' + str(data_betti) + "\n")
file.write('# of correspondence: ' + str(score) + "\n")
file.close()
