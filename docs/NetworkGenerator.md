# NetworkGenerator

This class provides method to serialize the creation of Keras + Tensorflow network

## Class Variables

## Constructors

## Methods

```python
build_model(depth=1,input_shape=(2,),width=8,activation='relu'):
  return tensorflow.keras.Sequential
```
Build a Keras + Tensorflow network with the given parameters. Parameters must be keras-compatible see [keras documentation](https://keras.io/api/)

---

```python
train_and_save(model=None, epoch_number=100, data= None, label=None, callbacks=None, batch_size=10, loss='sparse_categorical_crossentropy', save_path='./models.h5'):
```
Train a network and save it parameters are those of Keras, more information on its [documentation](https://keras.io/api/)

---

```python
full_net_combined(depth, input_shape, save_path, epoch_number, data, label):
```
Executes `build_model` then `train_and_save` with the given parameters, `batch_size = 64` and `loss='binary_crossentropy'`

