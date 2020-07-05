import sys

import numpy as np
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import datasets, layers, models

EPOCHS = 5
NUM_CLASSES = 10

# CIFAR_10 is a set of 60K images 32x32 pixels on 3 channels
IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32
BATCH_SIZE = 128


def build_model(input_shape, classes):
    model = models.Sequential()
    # 1st block
    model.add(layers.Convolution2D(32, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Convolution2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))
    # 2nd block
    model.add(layers.Convolution2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Convolution2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3))
    # 3d block
    model.add(layers.Convolution2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Convolution2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.4))
    # dense
    model.add(layers.Flatten())
    model.add(layers.Dense(classes, activation='softmax'))
    model.summary()
    return model


def load_data():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # normalize
    mean = np.mean(x_train, axis=(0, 1, 2, 3))
    std = np.std(x_train, axis=(0, 1, 2, 3))
    x_train = (x_train - mean) / (std + 1e-7)
    x_test = (x_test - mean) / (std + 1e-7)
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)
    return x_train, y_train, x_test, y_test


def func():
    (x_train, y_train, x_test, y_test) = load_data()
    model = build_model(input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS), classes=NUM_CLASSES)
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

    # Manipulate images to have more training data
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
    )
    datagen.fit(x_train)

    model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                        epochs=EPOCHS, verbose=1, validation_data=(x_test, y_test))
    # model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_test, y_test))

    # save model to disk
    model_json = model.to_json()
    with open('model.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights('model.h5')

    score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
    print("\nTest score:", score[0])
    print('Test accuracy:', score[1])


if __name__ == "__main__":
    func()
    sys.exit()
