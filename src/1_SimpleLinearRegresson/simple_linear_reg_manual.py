import sys

import matplotlib.pyplot as plt
import numpy as np

from common.LoadUtil import LoadUtil
from common.PlotUtil import PlotUtil


def func():
    x_train, x_test, y_train, y_test = LoadUtil.load_simple_linear_data_sk('studentscores.csv')

    PlotUtil.display_x_y(x_train, y_train, 'Hour', 'Score')

    W = sum(y_train * (x_train - np.mean(x_train))) / sum((x_train - np.mean(x_train)) ** 2)
    b = np.mean(y_train) - W * np.mean(x_train)
    print("The regression coefficients are ", W, b)

    y_pred = W * x_test + b

    PlotUtil.compare_x_y(x_test, y_test, y_pred, 'Hour', 'Score')


if __name__ == "__main__":
    try:
        plt.interactive(False)
        func()
    except Exception as e:
        print(e)
        sys.exit()
