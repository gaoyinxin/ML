import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

from common.LoadUtil import LoadUtil
from common.PlotUtil import PlotUtil


def func():
    ct = ColumnTransformer(
        ([
            ("OneHot", OneHotEncoder(), [3])
        ]), remainder='passthrough')

    x_train, x_test, y_train, y_test = LoadUtil.load_data_sk('50_Startups.csv', col_transformers=[ct])

    model = LinearRegression()
    model = model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    PlotUtil.compare_y(y_test, y_pred, x_label='#', y_label='Profit')

    print('intercept {}'.format(model.intercept_))
    print('coef {}'.format(model.coef_))
    print('y_pred = {}'.format(y_pred))
    print('MSE: {}'.format(metrics.mean_squared_error(y_test, y_pred)))
    print('RMSE: {}'.format(np.sqrt(metrics.mean_squared_error(y_test, y_pred))))

    cdf = pd.DataFrame(model.coef_, ['state1', 'state2', 'R&D Spend', 'Administration', 'Marketing Spend'],
                       columns=['Coefficients'])
    print(cdf)


if __name__ == "__main__":
    try:
        plt.interactive(True)
        func()
    except Exception as e:
        print(e)
        sys.exit()
