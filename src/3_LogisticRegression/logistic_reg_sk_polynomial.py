import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

from common.PlotUtil import PlotUtil


def PolynomialLogisticRegression(degree):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('stand_scalor', StandardScaler()),
        ('log_reg', LogisticRegression())
    ])


def func():
    dataset = pd.read_csv('../../resource/Social_Network_Ads.csv')
    x = dataset.iloc[:, [2, 3]].values
    y = dataset.iloc[:, 4].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    classifier = PolynomialLogisticRegression(5)
    classifier.fit(x_train, y_train)
    print(classifier.score(x_train, y_train))
    print(classifier.score(x_test, y_test))

    y_pred = classifier.predict(x_test)

    PlotUtil.display_confusion_matrix(y_test, y_pred)
    PlotUtil.display_decision_boundary(x_test, y_test, classifier)


if __name__ == "__main__":
    plt.interactive(True)
    func()
    sys.exit()
