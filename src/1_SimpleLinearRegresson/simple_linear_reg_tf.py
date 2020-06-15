import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    dataset = pd.read_csv('../../resource/studentscores.csv')
    X = dataset.iloc[:, 0].values
    Y = dataset.iloc[:, 1].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

    layer0 = tf.keras.layers.Dense(units=1, input_shape=[1])
    model = tf.keras.Sequential([layer0])

    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.5))
    history = model.fit(X_train, Y_train, epochs=100, verbose=False)

    weights = layer0.get_weights()
    print('weight: {} bias: {}'.format(weights[0], weights[1]))

    Y_pred = model.predict(X_test)

    plt.scatter(X_test, Y_test, color='red')
    plt.plot(X_test, Y_pred, color='blue')
    plt.show()

    plt.xlabel('Epoch Number')
    plt.ylabel("Loss Magnitude")
    plt.plot(history.history['loss'])
    plt.show()

