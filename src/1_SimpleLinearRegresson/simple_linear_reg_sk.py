import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    dataset = pd.read_csv('../../resource/studentscores.csv')
    X = dataset.iloc[:, : 1].values
    Y = dataset.iloc[:, 1].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

    regressor = LinearRegression()
    regressor = regressor.fit(X_train, Y_train)

    Y_pred = regressor.predict(X_test)

    plt.scatter(X_test, Y_test, color='red')
    plt.plot(X_test, Y_pred, color='blue')
    plt.show()
