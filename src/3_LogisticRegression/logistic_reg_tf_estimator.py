import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.feature_column as fc

from common.LoadUtil import LoadUtil
from common.PlotUtil import PlotUtil


def func():
    train_df, test_df = LoadUtil.load_data_df_sk('Social_Network_Ads.csv')
    x_train_df = train_df.iloc[:, 1:4]
    y_train_df = train_df.iloc[:, 4]
    x_test_df = test_df.iloc[:, 1:4]
    y_test_df = test_df.iloc[:, 4]

    PlotUtil.pairplot(x_train_df, hue='Gender')

    feature_columns = [
        fc.categorical_column_with_vocabulary_list('Gender', vocabulary_list=['Male', 'Female']),
        fc.numeric_column('Age', dtype=tf.float32, normalizer_fn=lambda x: (x / np.float32(100))),
        fc.numeric_column('EstimatedSalary', dtype=tf.float32, normalizer_fn=lambda x: (x / np.float32(100000)))]

    classifier = tf.estimator.LinearClassifier(feature_columns=feature_columns, n_classes=2, model_dir='logs/')

    train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
        x=x_train_df,
        y=y_train_df,
        num_epochs=None,
        shuffle=True)

    classifier.train(train_input_fn, steps=1000)

    test_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
        x=x_test_df,
        y=y_test_df,
        num_epochs=1,
        shuffle=False)

    y_pred = list()
    probabilities = classifier.predict(input_fn=test_input_fn)
    for i in range(len(y_test_df.values)):
        ret = next(probabilities)
        classes = ret['classes']
        y_pred.append(int(classes))
        print('test: {}  predict: {}'.format(y_test_df.values[i], classes))

    eval_results = classifier.evaluate(input_fn=test_input_fn)
    for key, value in sorted(eval_results.items()):
        print('%s: %s' % (key, value))

    PlotUtil.display_confusion_matrix(y_test_df.values, y_pred)


if __name__ == "__main__":
    plt.interactive(True)
    func()
    sys.exit()
