import os

import pandas as pd
from sklearn.model_selection import train_test_split


class LoadUtil:
    @staticmethod
    def load_simple_linear_data_sk(filename, train_ratio=0.8):
        dataset = pd.read_csv(LoadUtil.resource_path(filename))
        x = dataset.iloc[:, 0].values
        y = dataset.iloc[:, 1].values
        return train_test_split(x, y, test_size=1 - train_ratio, random_state=0)

    @staticmethod
    def load_data_sk(filename, col_transformers=None, train_ratio=0.8):
        dataset = pd.read_csv(LoadUtil.resource_path(filename))
        x = dataset.iloc[:, : -1].values
        y = dataset.iloc[:, -1].values
        if col_transformers is not None:
            for ct in col_transformers:
                x = ct.fit_transform(x)
                # Dummy variable trap
                x = x[:, 1:]
        return train_test_split(x, y, test_size=1 - train_ratio, random_state=0)

    @staticmethod
    def load_data_df_sk(filename, train_ratio=0.8):
        total_df = pd.read_csv(LoadUtil.resource_path(filename))
        return train_test_split(total_df, test_size=1 - train_ratio, random_state=0)

    @staticmethod
    def load_data(filename, train_ratio=0.8):
        dataset = pd.read_csv(LoadUtil.resource_path(filename))
        train_dataset = dataset.sample(frac=train_ratio, random_state=0)
        test_dataset = dataset.drop(train_dataset.index)
        x_train = train_dataset.iloc[:, :-1].values
        y_train = train_dataset.iloc[:, -1].values
        x_test = test_dataset.iloc[:, :-1].values
        y_test = test_dataset.iloc[:, -1].values
        return x_train, x_test, y_train, y_test

    @staticmethod
    def resource_path(filename):
        script_dir = os.path.split(os.path.realpath(__file__))[0]
        return script_dir + '/../../resource/' + filename
