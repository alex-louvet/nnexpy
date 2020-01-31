from generate_data import *
import numpy as np
from tensorflow import keras
from sklearn.utils import shuffle

dataDescriptor = DataDescriptor(nHoles=2, random=100, bounds=Bounds({
    'xmin': 0,
    'xmax': 1,
    'ymin': 0,
    'ymax': 1
}))
instance = dataDescriptor.generateData(classNumber=2)
instance.plot()
test = dataDescriptor.generateTestData(pointsNumber=10000)
data, label = instance.numpyify()
data, label = shuffle(data, label, random_state=0)

model = keras.Sequential([
    keras.layers.Dense(16, input_shape=(2,), activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])
model.summary()
model.compile(keras.optimizers.Adam(lr=.01),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(data, label, validation_split=0.2, batch_size=10,
          epochs=200, shuffle=True, verbose=2)
ùpredictedTest = test.predict(model, verbose=0)
predictedTest.plot()
