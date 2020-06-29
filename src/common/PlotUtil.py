import matplotlib.pyplot as plt


class PlotUtil:
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
    def compare_x_y(x_test, y_test, y_pred, x_label, y_label):
        plt.scatter(x_test, y_test, color='blue', label="Test Data")
        plt.plot(x_test, y_pred, color='red', label="Predicted Data")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.show()

    @staticmethod
    def compare_y(y_test, y_pred, x_label, y_label):
        plt.plot(y_test, color='blue', label="Test Data")
        plt.plot(y_pred, color='red', label="Predicted Data")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.show()
