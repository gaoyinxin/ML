import matplotlib.pyplot as plt

import seaborn as sns


class PlotUtil:
    @staticmethod
    def pairplot(df, hue=None):
        sns.set_style('white')
        sns.pairplot(df,
                     hue=hue,
                     kind='reg',  # 散点图/回归分布图{'scatter', 'reg'})
                     diag_kind='kde',  # 直方图/密度图{'hist'，
                     palette='husl')  # 图标大小
        plt.show()

    @staticmethod
    def display_x_y(x_test, y_test, x_label, y_label):
        plt.scatter(x_test, y_test, color='blue', label="Test Data")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()

    @staticmethod
    def display_loss(history):
        plt.xlabel('Epoch Number')
        plt.ylabel("Loss Magnitude")
        plt.plot(history.history['loss'])
        plt.show()

    @staticmethod
    def display_simple_linear_result(x_test, y_test, y_pred, x_label, y_label):
        plt.scatter(x_test, y_test, color='blue', label="Test Data")
        plt.plot(x_test, y_pred, color='red', label="Predicted Data")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.show()

    @staticmethod
    def display_multiple_linear_result(y_test, y_pred, x_label, y_label):
        plt.plot(y_test, color='blue', label="Test Data")
        plt.plot(y_pred, color='red', label="Predicted Data")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.show()
