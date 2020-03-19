import pickle
from generate_data import *
from tensorflow import keras
from os import walk, path
import time as t

score = [0, 0, 0, 0]

mypath = './models/'

with open(mypath + 'data_descriptor.pkl', 'rb') as input:
    centerList = pickle.load(input)
    radiusList = pickle.load(input)
    bounds = pickle.load(input)
    randomSeed = pickle.load(input)

dataDescriptor = DataDescriptor(nHoles=len(centerList), centerList=centerList,
                                radiusList=radiusList, random=t.time(), bounds=bounds)

instance = dataDescriptor.generateData(classNumber=2, pointsNumber=50000)
data_betti = instance.bettiNumbers(nPoints=5000)
test = dataDescriptor.generateTestData(pointsNumber=50000)

for directory in [x[0] for x in walk(mypath)]:

    model1 = keras.models.load_model(directory + '1layer.h5')
    model2 = keras.models.load_model(directory + '2layers.h5')
    model4 = keras.models.load_model(directory + '4layers.h5')
    model8 = keras.models.load_model(directory + '8layers.h5')

    predictedTest = test.predict(model1, verbose=0)
    if predictedTest.bettiNumbers(nPoints=10000) == data_betti:
        score[0] += 1

    predictedTest = test.predict(model2, verbose=0)
    if predictedTest.bettiNumbers(nPoints=10000) == data_betti:
        score[1] += 1

    predictedTest = test.predict(model4, verbose=0)
    if predictedTest.bettiNumbers(nPoints=10000) == data_betti:
        score[2] += 1

    predictedTest = test.predict(model8, verbose=0)
    if predictedTest.bettiNumbers(nPoints=10000) == data_betti:
        score[3] += 1

print(score)
