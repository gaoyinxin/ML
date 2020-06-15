import sys

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


def func():
    dataset = pd.read_csv('../../resource/studentscores.csv')
    x = dataset.iloc[:, 0].values
    y = dataset.iloc[:, 1].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    layer0 = tf.keras.layers.Dense(units=1, input_shape=[1])
    model = tf.keras.Sequential([layer0])

    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.5))
    history = model.fit(x_train, y_train, epochs=100, verbose=False)

    weights = layer0.get_weights()
    print('weight: {} bias: {}'.format(weights[0], weights[1]))

    y_pred = model.predict(x_test)

    plt.scatter(x_test, y_test, color='red')
    plt.plot(x_test, y_pred, color='blue')
    plt.show()

    plt.xlabel('Epoch Number')
    plt.ylabel("Loss Magnitude")
    plt.plot(history.history['loss'])
    plt.show()


if __name__ == "__main__":
    try:
        func()
    except Exception:
        sys.exit()
