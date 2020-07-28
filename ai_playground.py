from generate_data_dimension import *
import numpy as np
from tensorflow import keras
from sklearn.utils import shuffle
from generate_network import *
import pickle
import os
import sys
import shutil

randomSeed = 468643654

dataDescriptorList = [
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

        model1 = build_model(depth=1, input_shape=input_shape, width=8,
                             output_dimension=2, activation='relu')

        model2 = build_model(depth=2, input_shape=input_shape, width=8,
                             output_dimension=2, activation='relu')

        model4 = build_model(depth=4, input_shape=input_shape, width=8,
                             output_dimension=2, activation='relu')

        model6 = build_model(depth=6, input_shape=input_shape, width=8,
                             output_dimension=2, activation='relu')

        model8 = build_model(depth=8, input_shape=input_shape, width=8,
                             output_dimension=2, activation='relu')

        model10 = build_model(depth=10, input_shape=input_shape, width=8,
                              output_dimension=2, activation='relu')

        model12 = build_model(depth=12, input_shape=input_shape, width=8,
                              output_dimension=2, activation='relu')

        model14 = build_model(depth=14, input_shape=input_shape, width=8,
                              output_dimension=2, activation='relu')

        model16 = build_model(depth=16, input_shape=input_shape, width=8,
                              output_dimension=2, activation='relu')

        csv_logger = keras.callbacks.CSVLogger(
            mypath + '1layer.csv', separator=',', append=False)
        train_and_save(model=model1, epoch_number=epoch_number, data=data,
                       label=label, save_path=mypath + '1layer.h5', batch_size=64, loss="binary_crossentropy", callbacks=[csv_logger])

        csv_logger = keras.callbacks.CSVLogger(
            mypath + '2layers.csv', separator=',', append=False)
        train_and_save(model=model2, epoch_number=epoch_number, data=data,
                       label=label, save_path=mypath + '2layers.h5', batch_size=64, loss="binary_crossentropy", callbacks=[csv_logger])

        csv_logger = keras.callbacks.CSVLogger(
            mypath + '4layers.csv', separator=',', append=False)
        train_and_save(model=model4, epoch_number=epoch_number, data=data,
                       label=label, save_path=mypath + '4layers.h5', batch_size=64, loss="binary_crossentropy", callbacks=[csv_logger])

        csv_logger = keras.callbacks.CSVLogger(
            mypath + '6layers.csv', separator=',', append=False)
        train_and_save(model=model6, epoch_number=epoch_number, data=data,
                       label=label, save_path=mypath + '6layers.h5', batch_size=64, loss="binary_crossentropy", callbacks=[csv_logger])

        csv_logger = keras.callbacks.CSVLogger(
            mypath + '8layers.csv', separator=',', append=False)
        train_and_save(model=model8, epoch_number=epoch_number, data=data,
                       label=label, save_path=mypath + '8layers.h5', batch_size=64, loss="binary_crossentropy", callbacks=[csv_logger])

        csv_logger = keras.callbacks.CSVLogger(
            mypath + '10layers.csv', separator=',', append=False)
        train_and_save(model=model10, epoch_number=epoch_number, data=data,
                       label=label, save_path=mypath + '10layers.h5', batch_size=64, loss="binary_crossentropy", callbacks=[csv_logger])

        csv_logger = keras.callbacks.CSVLogger(
            mypath + '12layers.csv', separator=',', append=False)
        train_and_save(model=model12, epoch_number=epoch_number, data=data,
                       label=label, save_path=mypath + '12layers.h5', batch_size=64, loss="binary_crossentropy", callbacks=[csv_logger])

        csv_logger = keras.callbacks.CSVLogger(
            mypath + '14layers.csv', separator=',', append=False)
        train_and_save(model=model14, epoch_number=epoch_number, data=data,
                       label=label, save_path=mypath + '14layers.h5', batch_size=64, loss="binary_crossentropy", callbacks=[csv_logger])

        csv_logger = keras.callbacks.CSVLogger(
            mypath + '16layers.csv', separator=',', append=False)
        train_and_save(model=model8, epoch_number=epoch_number, data=data,
                       label=label, save_path=mypath + '16layers.h5', batch_size=64, loss="binary_crossentropy", callbacks=[csv_logger])
