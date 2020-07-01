import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import confusion_matrix


class PlotUtil:
    @staticmethod
    def display_decision_boundary(x, y, model, steps=1000, cmap='Paired'):
        """
        Function to plot the decision boundary and data points of a model.
        Data points are colored based on their actual label.
        """
        cmap = plt.get_cmap(cmap)

        # Define region of interest by data limits
        xmin, xmax = x[:, 0].min() - 1, x[:, 0].max() + 1
        ymin, ymax = x[:, 1].min() - 1, x[:, 1].max() + 1
        steps = 1000
        x_span = np.linspace(xmin, xmax, steps)
        y_span = np.linspace(ymin, ymax, steps)
        xx, yy = np.meshgrid(x_span, y_span)

        # Make predictions across region of interest
        labels = model.predict(np.c_[xx.ravel(), yy.ravel()])

        # Plot decision boundary in region of interest
        z = labels.reshape(xx.shape)

        fig, ax = plt.subplots()
        ax.contourf(xx, yy, z, cmap=cmap, alpha=0.5)

        ax.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap, lw=0)
        plt.show()

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
    def display_confusion_matrix(expected, predicted):
        cm = confusion_matrix(expected, predicted)
        sns.heatmap(cm, annot=True)
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
