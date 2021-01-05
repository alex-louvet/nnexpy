import pickle
from generate_data import *
import numpy as np
from tensorflow import keras
from sklearn.utils import shuffle
import pandas as pd
from os import walk, path
import matplotlib.pyplot as plt
import gudhi

# change path to visualize different models
mypath = "./models/D(0,2,0)/"

with open(mypath + 'data_descriptor.pkl', 'rb') as input:
    centerList = pickle.load(input)
    radiusList = pickle.load(input)
    bounds = pickle.load(input)
    randomSeed = pickle.load(input)

dataDescriptor = DataDescriptor(nHoles=len(centerList), centerList=centerList,
                                radiusList=radiusList, random=randomSeed, bounds=bounds)

instance = dataDescriptor.generateData(classNumber=2, pointsNumber=50000)
print(instance.bettiNumbers())
instance.plot()
instance_test = dataDescriptor.generateData(classNumber=2, pointsNumber=50000)
test = dataDescriptor.generateTestData(pointsNumber=50000)
data, label = instance.numpyify()
data, label = shuffle(data, label, random_state=0)

model1 = keras.models.load_model(mypath + '1layer.h5')
model2 = keras.models.load_model(mypath + '2layers.h5')
model4 = keras.models.load_model(mypath + '4layers.h5')
model8 = keras.models.load_model(mypath + '8layers.h5')

predictedTest = instance_test.predict(model1, verbose=0)
print(predictedTest.bettiNumbers(nPoints=10000))
predictedTest.plot()
predictedAccuracy = instance_test.predict_and_evaluate(model1, verbose=0)
print(predictedAccuracy)
predictedTest = test.predict(model2, verbose=0)
print(predictedTest.bettiNumbers(nPoints=10000))
predictedTest.plot()
predictedAccuracy = instance_test.predict_and_evaluate(model2, verbose=0)
print(predictedAccuracy)
predictedTest = test.predict(model4, verbose=0)
print(predictedTest.bettiNumbers(nPoints=10000))
predictedTest.plot()
predictedAccuracy = instance_test.predict_and_evaluate(model4, verbose=0)
print(predictedAccuracy)
predictedTest = test.predict(model8, verbose=0)
print(predictedTest.bettiNumbers(nPoints=10000))
predictedTest.plot()
predictedAccuracy = instance_test.predict_and_evaluate(model8, verbose=0)
print(predictedAccuracy)

f = []
for (dirpath, dirnames, filenames) in walk(mypath):
    f.extend(filenames)
    break

for file in f:
    if path.splitext(mypath + file)[1] == '.csv':
        csv = pd.read_csv(mypath + file)
        X = csv['epoch']
        Y = csv['val_accuracy']
        plt.plot(X, Y, label=file)
plt.legend()
plt.show()
