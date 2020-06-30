import sys

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.feature_column as fc

from common.LoadUtil import LoadUtil
from common.PlotUtil import PlotUtil


def func():
    train_df, test_df = LoadUtil.load_data_df_sk('50_Startups.csv')
    x_train_df = train_df.iloc[:, :4]
    y_train_df = train_df.iloc[:, 4]
    x_test_df = test_df.iloc[:, :4]
    y_test_df = test_df.iloc[:, 4]

    PlotUtil.pairplot(x_train_df)

    feature_columns = [fc.numeric_column('RnDSpend', dtype=tf.float32),
                       fc.numeric_column('Administration', dtype=tf.float32),
                       fc.numeric_column('MarketingSpend', dtype=tf.float32),
                       fc.categorical_column_with_vocabulary_list('State', vocabulary_list=['New York', 'California',
                                                                                            'Florida'])]

    train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
        x=x_train_df,
        y=y_train_df,
        num_epochs=None,
        shuffle=True)

    linear_est = tf.estimator.LinearRegressor(feature_columns=feature_columns, model_dir='logs/')
    linear_est.train(train_input_fn, steps=100)

    test_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
        x=x_test_df,
        y=y_test_df,
        num_epochs=1,
        shuffle=False)
    predictions = linear_est.predict(test_input_fn)

    y_pred = list()
    for i in range(len(y_test_df.values)):
        predict = next(predictions)['predictions']
        y_pred.append(predict)

    PlotUtil.display_multiple_linear_result(y_test_df.values, y_pred, x_label='#', y_label='Profit')


if __name__ == "__main__":
    plt.interactive(True)
    func()
    sys.exit()
