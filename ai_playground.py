from generate_data import *
import numpy as np
from tensorflow import keras
from sklearn.utils import shuffle
from generate_network import *
import pickle
import os
import sys

randomSeed = 468643654

myRootPath = "./models2/"

dataDescriptor = DataDescriptor(nHoles=2, centerList=[DataPoint({"x": 0.25, "y": 0.25}), DataPoint({"x": 0.75, "y": 0.75})], radiusList=[[(0, 0.05), (0.15, 0.16)], [(0, 0.05), (0.15, 0.16)]], random=randomSeed, bounds=Bounds({
    'xmin': 0,
    'xmax': 1,
    'ymin': 0,
    'ymax': 1
}))

with open(myRootPath + 'data_descriptor.pkl', 'wb') as output:
    pickle.dump(dataDescriptor.centerList, output, pickle.HIGHEST_PROTOCOL)
    pickle.dump(dataDescriptor.radiusList, output, pickle.HIGHEST_PROTOCOL)
    pickle.dump(dataDescriptor.bounds, output, pickle.HIGHEST_PROTOCOL)
    pickle.dump(randomSeed, output, pickle.HIGHEST_PROTOCOL)

instance = dataDescriptor.generateData(classNumber=2, pointsNumber=10000)
data, label = instance.numpyify()
data, label = shuffle(data, label, random_state=0)

iterNum = 1
epoch_number = 1000
startIter = 0
if (len(sys.argv) > 1):
    iterNum = int(sys.argv[1])

if (len(sys.argv) > 2):
    epoch_number = int(sys.argv[2])

if (len(sys.argv) > 3):
    startIter = int(sys.argv[3])

file = open(myRootPath + 'information.txt', 'w')
file.write('Iteration Number: ' + str(iterNum))
file.write('Epoch Number: ' + str(epoch_number))
file.close()

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

    model1 = build_model(depth=1, input_shape=(2,), width=8,
                         output_dimension=2, activation='relu')

    model2 = build_model(depth=2, input_shape=(2,), width=8,
                         output_dimension=2, activation='relu')

    model4 = build_model(depth=4, input_shape=(2,), width=8,
                         output_dimension=2, activation='relu')

    model8 = build_model(depth=8, input_shape=(2,), width=8,
                         output_dimension=2, activation='relu')

    model16 = build_model(depth=16, input_shape=(2,), width=8,
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
        mypath + '8layers.csv', separator=',', append=False)
    train_and_save(model=model8, epoch_number=epoch_number, data=data,
                   label=label, save_path=mypath + '8layers.h5', batch_size=64, loss="binary_crossentropy", callbacks=[csv_logger])

    csv_logger = keras.callbacks.CSVLogger(
        mypath + '16layers.csv', separator=',', append=False)
    train_and_save(model=model8, epoch_number=epoch_number, data=data,
                   label=label, save_path=mypath + '16layers.h5', batch_size=64, loss="binary_crossentropy", callbacks=[csv_logger])
