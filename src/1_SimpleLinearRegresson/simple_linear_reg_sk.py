import sys

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from common.LoadUtil import LoadUtil
from common.PlotUtil import PlotUtil


def func():
    x_train, x_test, y_train, y_test = LoadUtil.load_data_sk('studentscores.csv')

    model = LinearRegression()
    model = model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    PlotUtil.compare_x_y(x_test, y_test, y_pred, 'Hour', 'Score')


if __name__ == "__main__":
    try:
        plt.interactive(True)
        func()
    except Exception as e:
        print(e)
        sys.exit()
