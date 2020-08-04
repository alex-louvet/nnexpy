import shutil
import sys
import os
import pickle
from sklearn.utils import shuffle
from generate_data_dimension import *
import numpy as np
import gc
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

randomSeed = 468643654

dataDescriptorList = [
    DataDescriptor(dimension=2, nHoles=2, centerList=[DataPoint(coordinates=(0.24, 0.25)), DataPoint(
        coordinates=(0.75, 0.75))], radiusList=[[(0, 0.1)], [(0.1, 0.13)]], holeDimension=[2, 1]),

    DataDescriptor(dimension=2, nHoles=2, centerList=[DataPoint(coordinates=(0.24, 0.25)), DataPoint(
        coordinates=(0.74, 0.75))], radiusList=[[(0, 0.1)], [(0.1, 0.13)]], holeDimension=[2, 2]),

    DataDescriptor(dimension=2, nHoles=1, centerList=[DataPoint(coordinates=(
        0.51, 0.49))], radiusList=[[(0.1, 0.13)]], holeDimension=[2]),

    DataDescriptor(dimension=2, nHoles=1, centerList=[DataPoint(coordinates=(
        0.51, 0.49))], radiusList=[[(0.1, 0.13)]], holeDimension=[1]),

    DataDescriptor(dimension=3, nHoles=2, centerList=[DataPoint(coordinates=(0.25, 0.24, 0.25)), DataPoint(
        coordinates=(0.74, 0.75, 0.75))], radiusList=[[(0, 0.1)], [(0.1, 0.13)]], holeDimension=[3, 2]),

    DataDescriptor(dimension=3, nHoles=2, centerList=[DataPoint(coordinates=(0.25, 0.24, 0.25)), DataPoint(
        coordinates=(0.74, 0.75, 0.75))], radiusList=[[(0, 0.1)], [(0.1, 0.13)]], holeDimension=[3, 3]),

    DataDescriptor(dimension=3, nHoles=1, centerList=[DataPoint(coordinates=(
        0.51, 0.49, 0.5))], radiusList=[[(0.1, 0.13)]], holeDimension=[3]),

    DataDescriptor(dimension=3, nHoles=1, centerList=[DataPoint(coordinates=(
        0.51, 0.49, 0.5))], radiusList=[[(0.1, 0.13)]], holeDimension=[2]),

    DataDescriptor(dimension=4, nHoles=2, centerList=[DataPoint(coordinates=(0.25, 0.24, 0.25, 0.26)), DataPoint(
        coordinates=(0.74, 0.75, 0.75, 0.76))], radiusList=[[(0, 0.1)], [(0.1, 0.13)]], holeDimension=[4, 3]),

    DataDescriptor(dimension=4, nHoles=2, centerList=[DataPoint(coordinates=(0.25, 0.24, 0.25, 0.26)), DataPoint(
        coordinates=(0.74, 0.75, 0.75, 0.76))], radiusList=[[(0, 0.1)], [(0.1, 0.13)]], holeDimension=[4, 4]),

    DataDescriptor(dimension=4, nHoles=1, centerList=[DataPoint(coordinates=(
        0.51, 0.49, 0.5, 0.5))], radiusList=[[(0.1, 0.13)]], holeDimension=[4]),

    DataDescriptor(dimension=4, nHoles=1, centerList=[DataPoint(coordinates=(
        0.51, 0.49, 0.5, 0.5))], radiusList=[[(0.1, 0.13)]], holeDimension=[3]),
]

iterNum = 1
epoch_number = 1000
startIter = 0
if (len(sys.argv) > 1):
    iterNum = int(sys.argv[1])

if (len(sys.argv) > 2):
    epoch_number = int(sys.argv[2])

if (len(sys.argv) > 3):
    startIter = int(sys.argv[3])

for j, x in enumerate(dataDescriptorList):

    myRootPath = "./models/instance_" + str(j) + "/"

    if not os.path.exists(myRootPath):
        os.makedirs(myRootPath)
    else:
        for root, dirs, files in os.walk(myRootPath):
            for file in files:
                os.remove(os.path.join(root, file))

    dataDescriptor = x

    with open(myRootPath + 'data_descriptor.pkl', 'wb') as output:
        pickle.dump(dataDescriptor.centerList, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(dataDescriptor.radiusList, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(dataDescriptor.bounds, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(randomSeed, output, pickle.HIGHEST_PROTOCOL)

    instance = dataDescriptor.generateData(classNumber=2, pointsNumber=10000)
    data, label = instance.numpyify()
    data, label = shuffle(data, label, random_state=0)

    file = open(myRootPath + 'information.txt', 'w')
    file.write('Iteration Number: ' + str(iterNum) + "\n")
    file.write('Epoch Number: ' + str(epoch_number) + "\n")
    file.close()

    input_shape = (len(dataDescriptor.centerList[0].coordinates),)
    for i in range(startIter, startIter + iterNum):
        mypath = myRootPath + 'training_' + str(i) + '/'
        print('\n\n\n################################################# ' +
              str(i) + ' ##########################################\n\n\n')
        if not os.path.exists(mypath):
            os.makedirs(mypath)
        else:
            for root, dirs, files in os.walk(mypath):
                for file in files:
                    os.remove(os.path.join(root, file))

        for i in [1, 2, 4, 6, 8, 10, 12, 14, 16]:

            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Dense(
                8, input_dim=input_shape[0], activation='relu', kernel_initializer='he_uniform'))
            for _ in range(i):
                model.add(tf.keras.layers.Dense(8, activation='relu'))
            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

            csv = tf.keras.callbacks.CSVLogger(
                mypath + str(i) + 'layer.csv', separator=',', append=False)
            model.summary()
            model.compile(optimizer="adam",
                          loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(data, label, validation_split=0.2, batch_size=64,
                      epochs=epoch_number, shuffle=True, verbose=2, callbacks=[csv])
            model.save(mypath + str(i) + 'layer.h5')
            del model
            del csv
            gc.collect()
            tf.keras.backend.clear_session()
            tf.compat.v1.reset_default_graph()
