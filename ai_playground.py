from generate_data import *
import numpy as np
from tensorflow import keras
from sklearn.utils import shuffle

dataDescriptor = DataDescriptor(nHoles=3, maxStrata=2, random=64864518, bounds=Bounds({
    'xmin': 0,
    'xmax': 1,
    'ymin': 0,
    'ymax': 1
}))
instance = dataDescriptor.generateData(classNumber=2, pointsNumber=5000)
instance.plot()
test = dataDescriptor.generateTestData(pointsNumber=10000)
data, label = instance.numpyify()
data, label = shuffle(data, label, random_state=0)

model1 = keras.Sequential([
    keras.layers.Dense(8, input_shape=(2,), activation='relu'),
    keras.layers.Dense(4, activation='softmax')
])
model2 = keras.Sequential([
    keras.layers.Dense(8, input_shape=(2,), activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(4, activation='softmax')
])
model3 = keras.Sequential([
    keras.layers.Dense(8, input_shape=(2,), activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(4, activation='softmax')
])
model4 = keras.Sequential([
    keras.layers.Dense(8, input_shape=(2,), activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(4, activation='softmax')
])

model1.summary()
model1.compile(keras.optimizers.Adam(lr=.01),
               loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model1.fit(data, label, validation_split=0.2, batch_size=10,
           epochs=20, shuffle=True, verbose=2)
predictedTest = test.predict(model1, verbose=0)
predictedTest.plot()
model2.summary()
model2.compile(keras.optimizers.Adam(lr=.01),
               loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model2.fit(data, label, validation_split=0.2, batch_size=10,
           epochs=20, shuffle=True, verbose=2)
predictedTest = test.predict(model2, verbose=0)
predictedTest.plot()
model3.summary()
model3.compile(keras.optimizers.Adam(lr=.01),
               loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model3.fit(data, label, validation_split=0.2, batch_size=10,
           epochs=20, shuffle=True, verbose=2)
predictedTest = test.predict(model3, verbose=0)
predictedTest.plot()
model4.summary()
model4.compile(keras.optimizers.Adam(lr=.01),
               loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model4.fit(data, label, validation_split=0.2, batch_size=10,
           epochs=200, shuffle=True, verbose=2)
predictedTest = test.predict(model4, verbose=0)
predictedTest.plot()
