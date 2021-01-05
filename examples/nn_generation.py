def build_model(*args, **kwargs):
    import tensorflow as tf

    depth = kwargs.get('depth', 1)
    input_shape = kwargs.get('input_shape', (2,))
    width = kwargs.get('width', 8)
    activation = kwargs.get('activation', 'relu')

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(8, input_dim=input_shape[0], activation=activation,
                                    kernel_initializer='he_uniform'))
    for _ in range(depth):
        model.add(tf.keras.layers.Dense(8, activation=activation))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model


def train_and_save(*args, **kwargs):
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()

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
    import gc
    del model
    gc.collect()
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()


def full_net_combined(i, input_shape, mypath, epoch_number, data, label):
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    model = build_model(
        depth=i, input_shape=input_shape, width=8, activation='relu')
    csv = tf.keras.callbacks.CSVLogger(
        mypath + str(i) + 'layer.csv', separator=',', append=False)
    train_and_save(model=model, epoch_number=epoch_number, data=data, label=label, save_path=mypath +
                   str(i) + 'layer.h5', batch_size=64, loss="binary_crossentropy", callbacks=[csv])
    import gc
    del model
    gc.collect()
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
