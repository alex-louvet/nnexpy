def build_model(*args, **kwargs):
    from tensorflow import keras

    depth = kwargs.get('depth', 1)
    input_shape = kwargs.get('input_shape', (2,))
    width = kwargs.get('width', 8)
    output_dimension = kwargs.get('output_dimension', 2)
    activation = kwargs.get('activation', 'relu')

    model = keras.Sequential()

    model.add(keras.layers.Dense(8, input_dim=input_shape[0], activation=activation,
                                 kernel_initializer='he_uniform'))
    for _ in range(depth):
        model.add(keras.layers.Dense(8, activation=activation))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    return model


def train_and_save(*args, **kwargs):
    from tensorflow import keras

    model = kwargs.get('model', None)
    epoch_number = kwargs.get('epoch_number', 100)
    data = kwargs.get('data', None)
    label = kwargs.get('label', None)
    save_path = kwargs.get('save_path', './model.h5')
    callbacks = kwargs.get('callbacks', None)
    batch_size = kwargs.get('batch_size', 10)
    loss = kwargs.get('loss', 'sparse_categorical_crossentropy')

    model.summary()
    model.compile(optimizer="adam",
                  loss=loss, metrics=['accuracy'])
    model.fit(data, label, validation_split=0.2, batch_size=batch_size,
              epochs=epoch_number, shuffle=True, verbose=2, callbacks=callbacks)
    model.save(save_path)
