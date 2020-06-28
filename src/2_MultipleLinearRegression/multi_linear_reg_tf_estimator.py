import sys

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow.feature_column as fc
from sklearn.model_selection import train_test_split


def estimator_input_fn(df_data, df_result):
    def input_function():
        features_dict = df_data.to_dict(orient='list')
        result_array = df_result.to_dict(orient='list').get('Profit')
        return features_dict, result_array

    return input_function


def predict_input_fn(df_data):
    def input_function():
        return df_data.to_dict(orient='list')
    return input_function


def func():
    plt.interactive(True)

    dataset = pd.read_csv('../../resource/50_Startups.csv')
    # train_dataset = dataset.sample(frac=0.8, random_state=0)
    # test_dataset = dataset.drop(train_dataset.index)
    x = dataset.iloc[:, : -1].values
    y = dataset.iloc[:, -1].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
    # x_train = train_dataset.iloc[:, :-1].values
    # y_train = train_dataset.iloc[:, -1].values
    # x_test = test_dataset.iloc[:, :-1].values
    # y_test = test_dataset.iloc[:, -1].values
    print(x_test)

    features = ['RnDSpend', 'Administration', 'MarketingSpend', 'State']

    x_train_df = pd.DataFrame(x_train, columns=features)
    x_test_df = pd.DataFrame(x_test, columns=features)
    y_train_df = pd.DataFrame(y_train, columns=['Profit'])

    feature_columns = [fc.numeric_column('RnDSpend', dtype=tf.float32),
                       fc.numeric_column('Administration', dtype=tf.float32),
                       fc.numeric_column('MarketingSpend', dtype=tf.float32),
                       fc.categorical_column_with_vocabulary_list('State', vocabulary_list=['New York', 'California',
                                                                                            'Florida'])]

    train_input_fn = estimator_input_fn(x_train_df, y_train_df)
    test_input_fn = predict_input_fn(x_test_df)

    linear_est = tf.estimator.LinearRegressor(feature_columns=feature_columns)
    linear_est.train(train_input_fn, steps=100)
    predictions = linear_est.predict(test_input_fn)

    y_pred = list()
    for i in range(len(x_test)):
        predict = next(predictions)['predictions']
        y_pred.append(predict)

    plt.xlabel('test sample')
    plt.ylabel("value")
    plt.plot(y_test, color='blue')
    plt.plot(y_pred, color='red')
    plt.show()


if __name__ == "__main__":
    try:
        func()
    except Exception as e:
        print(e)
        sys.exit()
