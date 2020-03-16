from generate_data import *
import numpy as np
from tensorflow import keras
from sklearn.utils import shuffle
from generate_network import *

dataDescriptor = DataDescriptor(nHoles=2, maxStrata=2, random=11354445321, bounds=Bounds({
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

model16 = build_model(depth=16, input_shape=(2,), width=8,
                      output_dimension=2, activation='relu')

csv_logger = keras.callbacks.CSVLogger(
    './models/1layer2602csv.csv', separator=',', append=False)
train_and_save(model=model1, epoch_number=100, data=data,
               label=label, save_path='./models/1layer2602.h5', callbacks=[csv_logger])
"""
train_and_save(model=model2, epoch_number=100, data=data,
               label=label, save_path='./models/2layer2602.h5')
train_and_save(model=model4, epoch_number=100, data=data,
               label=label, save_path='./models/4layer2602.h5')
train_and_save(model=model8, epoch_number=100, data=data,
               label=label, save_path='./models/8layer2602.h5')
train_and_save(model=model16, epoch_number=100, data=data,
               label=label, save_path='./models/16layer2602.h5')
"""
