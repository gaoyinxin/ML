import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def func():
    plt.interactive(True)
    dataset = pd.read_csv('../../resource/studentscores.csv')
    x = dataset.iloc[:, : 1].values
    y = dataset.iloc[:, 1].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    model = LinearRegression()
    model = model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    plt.scatter(x_test, y_test, color='red')
    plt.plot(x_test, y_pred, color='blue')
    plt.show()


if __name__ == "__main__":
    try:
        func()
    except Exception as e:
        print(e)
        sys.exit()
