import sys

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

from common.PlotUtil import PlotUtil


def build_model(input_shape, classes):
    model = models.Sequential()
    model.add(layers.Convolution2D(20, (5, 5), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Convolution2D(50, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(500, activation='relu'))
    model.add(layers.Dense(classes, activation='softmax'))
    return model


def func():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    # reshape
    x_train = x_train.reshape((60000, 28, 28, 1))
    x_test = x_test.reshape((10000, 28, 28, 1))
    # normalize
    x_train, x_test = x_train / 255.0, x_test / 255.0
    # cast
    X_train = x_train.astype('float32')
    X_test = x_test.astype('float32')

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # initialize the optimizer and model
    model = build_model(input_shape=(28, 28, 1), classes=10)
    model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
    model.summary()

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir='./logs')
    ]

    # fit
    history = model.fit(X_train, y_train, batch_size=128, epochs=20, verbose=1,
                        validation_split=0.2, callbacks=callbacks)
    score = model.evaluate(X_test, y_test, verbose=1)
    print("\nTest score:", score[0])
    print('Test accuracy:', score[1])
    PlotUtil.display_loss(history)


if __name__ == "__main__":
    func()
    sys.exit()
