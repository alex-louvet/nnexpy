from generate_data import *
import numpy as np
from tensorflow import keras
from sklearn.utils import shuffle
from generate_network import *

dataDescriptor = DataDescriptor(nHoles=1, centerList=[DataPoint({"x": 0.5, "y": 0.5})], radiusList=[[(0, 0.05), (0.15, 0.16), (0.3, 0.31)]], random=468643654, bounds=Bounds({
    'xmin': 0,
    'xmax': 1,
    'ymin': 0,
    'ymax': 1
}))

instance = dataDescriptor.generateData(classNumber=2, pointsNumber=10000)
data, label = instance.numpyify()
data, label = shuffle(data, label, random_state=0)

model1 = build_model(depth=1, input_shape=(2,), width=8,
                     output_dimension=2, activation='relu')

model2 = build_model(depth=2, input_shape=(2,), width=8,
                     output_dimension=2, activation='relu')

model4 = build_model(depth=4, input_shape=(2,), width=8,
                     output_dimension=2, activation='relu')

model8 = build_model(depth=8, input_shape=(2,), width=8,
                     output_dimension=2, activation='relu')

csv_logger = keras.callbacks.CSVLogger(
    './models/1layer.csv', separator=',', append=False)
train_and_save(model=model1, epoch_number=200, data=data,
               label=label, save_path='./models/1layers.h5', batch_size=64, loss="binary_crossentropy", callbacks=[csv_logger])

csv_logger = keras.callbacks.CSVLogger(
    './models/2layers.csv', separator=',', append=False)
train_and_save(model=model2, epoch_number=200, data=data,
               label=label, save_path='./models/2layers.h5', batch_size=64, loss="binary_crossentropy", callbacks=[csv_logger])

csv_logger = keras.callbacks.CSVLogger(
    './models/4layers.csv', separator=',', append=False)
train_and_save(model=model4, epoch_number=200, data=data,
               label=label, save_path='./models/4layers.h5', batch_size=64, loss="binary_crossentropy", callbacks=[csv_logger])

csv_logger = keras.callbacks.CSVLogger(
    './models/8layers.csv', separator=',', append=False)
train_and_save(model=model8, epoch_number=200, data=data,
               label=label, save_path='./models/8layers.h5', batch_size=64, loss="binary_crossentropy", callbacks=[csv_logger])
