import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf


def get_feature_cols():
    feature_columns = []
    feature_layer_inputs = {}

    for header in ['R&D Spend', 'Administration', 'Marketing Spend']:
        feature_columns.append(tf.feature_column.numeric_column(header))

    thal = tf.feature_column.categorical_column_with_vocabulary_list('State', ['New York', 'California', 'Florida'])
    thal_one_hot = tf.feature_column.indicator_column(thal)
    feature_columns.append(thal_one_hot)

    return feature_columns


def func():
    plt.interactive(True)
    print("GPU Available: ", tf.test.is_gpu_available())

    dataset = pd.read_csv('../../resource/50_Startups.csv')
    # dataset.isna().sum()
    # dataset = dataset.dropna()
    # dataset.tail()

    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    x_train = train_dataset.iloc[:, :-1].values
    y_train = train_dataset.iloc[:, -1].values
    x_test = test_dataset.iloc[:, :-1].values
    y_test = test_dataset.iloc[:, -1].values

    # usage https://zhuanlan.zhihu.com/p/98729226
    sns.pairplot(train_dataset[['R&D Spend', 'Administration', 'Marketing Spend']], kind='reg', diag_kind="kde")
    # plt.show()

    # train_stats = train_dataset.describe()
    # train_stats = train_stats.transpose()
    # print(train_stats)

    feature_layer = tf.keras.layers.DenseFeatures(get_feature_cols())
    model = tf.keras.Sequential([feature_layer])

    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.5))
    history = model.fit(x_train, y_train, epochs=100, verbose=False)
    model.summary()

    weights = feature_layer.get_weights()
    print('weight: {} bias: {}'.format(weights[0], weights[1]))

    y_pred = model.predict(x_test)
    print('y_pred = {}'.format(y_pred))

    plt.xlabel('test sample')
    plt.ylabel("value")
    plt.plot(y_test, color='blue')
    plt.plot(y_pred, color='red')
    plt.show()

    plt.xlabel('Epoch Number')
    plt.ylabel("Loss Magnitude")
    plt.plot(history.history['loss'])
    plt.show()


if __name__ == "__main__":
    try:
        func()
    except Exception as e:
        print(e)
        sys.exit()
