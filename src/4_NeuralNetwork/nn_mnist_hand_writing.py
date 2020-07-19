import sys

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

from common.PlotUtil import PlotUtil


def func():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    # X_train is 60000 rows of 28x28 values; we reshape it to 60000 x 784.
    RESHAPED = 784
    #
    x_train = x_train.reshape(60000, RESHAPED)
    x_test = x_test.reshape(10000, RESHAPED)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Normalize inputs to be within in [0, 1].
    x_train, x_test = x_train / 255.0, x_test / 255.0
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Labels have one-hot representation.
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    # Build the model.
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(128, input_shape=(RESHAPED,), name='dense_layer', activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(128, name='dense_layer_2', activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(10, name='dense_layer_3', activation='softmax'))
    # Summary of the model.
    model.summary()
    # Compiling the model. If use RMSProp, we can converge faster, so epoch could be increased to 250
    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
    # Training the model.
    history = model.fit(x_train, y_train, batch_size=128, epochs=50, verbose=1, validation_split=0.2)
    # Evaluating the model.
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy:', test_acc)
    PlotUtil.display_loss(history)

    tf.keras.callbacks.TensorBoard(log_dir='./logs')


if __name__ == "__main__":
    func()
    sys.exit()
