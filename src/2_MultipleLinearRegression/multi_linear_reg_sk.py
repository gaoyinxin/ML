import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def func():
    plt.interactive(True)
    dataset = pd.read_csv('../../resource/50_Startups.csv')
    x = dataset.iloc[:, : -1].values
    y = dataset.iloc[:, -1].values

    ct = ColumnTransformer(
        ([
            ("OneHot", OneHotEncoder(), [3])
        ]), remainder='passthrough')

    x = ct.fit_transform(x)

    # Dummy variable trap
    x = x[:, 1:]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

    model = LinearRegression()
    model = model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    plt.xlabel('test sample')
    plt.ylabel("value")
    plt.plot(y_test, color='blue')
    plt.plot(y_pred, color='red')
    plt.show()

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
        func()
    except Exception as e:
        print(e)
        sys.exit()
