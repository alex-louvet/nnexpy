import pickle
import sys
import gc
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

i = int(sys.argv[1])
input_shape = int(sys.argv[2])
mypath = sys.argv[3]
epoch_number = int(sys.argv[4])


with open(sys.argv[5], 'rb') as input:
    data = pickle.load(input)

with open(sys.argv[6], 'rb') as input:
    label = pickle.load(input)

print(i, input_shape, epoch_number)

model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(
    8, input_dim=input_shape, activation='relu', kernel_initializer='he_uniform'))
for _ in range(i):
    model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

csv = tf.keras.callbacks.CSVLogger(
    mypath + str(i) + 'layer.csv', separator=',', append=False)

model.summary()
model.compile(optimizer="adam", loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(data, label, validation_split=0.2, batch_size=64,
          epochs=epoch_number, shuffle=True, verbose=2, callbacks=[csv])
model.save(mypath + str(i) + 'layer.h5')
del model
gc.collect()
tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()
