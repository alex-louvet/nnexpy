def build_model(*args, **kwargs):
    from tensorflow import keras

    depth = kwargs.get('depth', 1)
    input_shape = kwargs.get('input_shape', (2,))
    width = kwargs.get('width', 8)
    output_dimension = kwargs.get('output_dimension', 2)
    activation = kwargs.get('activation', 'relu')

    model = keras.Sequential([
        keras.layers.Dense(width, input_shape=input_shape,
                           activation=activation)
    ])
    for _ in range(depth - 1):
        model.add(keras.layers.Dense(width, activation=activation))
    model.add(keras.layers.Dense(output_dimension, activation='softmax'))
    return model


def train_and_save(*args, **kwargs):
    from tensorflow import keras

    model = kwargs.get('model', None)
    epoch_number = kwargs.get('epoch_number', 100)
    data = kwargs.get('data', None)
    label = kwargs.get('label', None)
    save_path = kwargs.get('save_path', './model.h5')

    model.summary()
    model.compile(keras.optimizers.Adam(lr=.01),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(data, label, validation_split=0.2, batch_size=10,
              epochs=epoch_number, shuffle=True, verbose=2)
    model.save(save_path)
