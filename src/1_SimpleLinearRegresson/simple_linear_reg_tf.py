import sys

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf


def func():
    plt.interactive(True)
    dataset = pd.read_csv('../../resource/studentscores.csv')
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    x_train = train_dataset.iloc[:, 0].values
    y_train = train_dataset.iloc[:, 1].values
    x_test = test_dataset.iloc[:, 0].values
    y_test = test_dataset.iloc[:, 1].values

    layer0 = tf.keras.layers.Dense(units=1, input_shape=[1])
    model = tf.keras.Sequential([layer0])

    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.5), metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=100, verbose=False)

    weights = layer0.get_weights()
    print('weight: {} bias: {}'.format(weights[0], weights[1]))

    y_pred = model.predict(x_test)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy: {}'.format(test_acc))

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
    except Exception as e:
        print(e)
        sys.exit()
