import sys

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow.feature_column as fc

from common.LoadUtil import LoadUtil
from common.PlotUtil import PlotUtil


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
    x_train, x_test, y_train, y_test = LoadUtil.load_data_sk('50_Startups.csv')

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

    linear_est = tf.estimator.LinearRegressor(feature_columns=feature_columns, model_dir='logs/')
    linear_est.train(train_input_fn, steps=100)
    predictions = linear_est.predict(test_input_fn)

    y_pred = list()
    for i in range(len(x_test)):
        predict = next(predictions)['predictions']
        y_pred.append(predict)

    PlotUtil.compare_y(y_test, y_pred, x_label='#', y_label='Profit')


if __name__ == "__main__":
    try:
        plt.interactive(True)
        func()
    except Exception as e:
        print(e)
        sys.exit()
