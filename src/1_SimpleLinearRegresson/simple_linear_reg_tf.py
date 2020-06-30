import sys

import matplotlib.pyplot as plt
import tensorflow as tf

from common.LoadUtil import LoadUtil
from common.PlotUtil import PlotUtil


def func():
    x_train, x_test, y_train, y_test = LoadUtil.load_data_sk('studentscores.csv')

    layer0 = tf.keras.layers.Dense(units=1, input_shape=[1])
    model = tf.keras.Sequential([layer0])

    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.5), metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=100, verbose=False)

    weights = layer0.get_weights()
    print('weight: {} bias: {}'.format(weights[0], weights[1]))

    y_pred = model.predict(x_test)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy: {}'.format(test_acc))

    PlotUtil.display_simple_linear_result(x_test, y_test, y_pred, 'Hour', 'Score')
    PlotUtil.display_loss(history)


if __name__ == "__main__":
    plt.interactive(True)
    func()
    sys.exit()
