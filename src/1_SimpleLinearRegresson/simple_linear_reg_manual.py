import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def func():
    plt.interactive(False)
    dataset = pd.read_csv('../../resource/studentscores.csv')
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    x_train = train_dataset.iloc[:, 0].values
    y_train = train_dataset.iloc[:, 1].values
    x_test = test_dataset.iloc[:, 0].values
    y_test = test_dataset.iloc[:, 1].values

    plt.scatter(x_train, y_train)
    plt.show()

    W = sum(y_train * (x_train - np.mean(x_train))) / sum((x_train - np.mean(x_train)) ** 2)
    b = np.mean(y_train) - W * np.mean(x_train)
    print("The regression coefficients are ", W, b)

    y_pred = W * x_test + b

    plt.plot(x_test, y_pred, color='red', label="Predicted Data")
    plt.scatter(x_test, y_test, label="Training Data")
    plt.xlabel('Hour')
    plt.ylabel("Score")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    try:
        func()
    except Exception as e:
        print(e)
        sys.exit()
