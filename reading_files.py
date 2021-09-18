import pandas as pd


def read_files(path_train, path_test):
    train = pd.read_csv(path_train)
    train.drop(['id', 'keyword', 'location'], axis=1, inplace=True)

    test = pd.read_csv(path_test)
    test.drop(['keyword', 'location'], axis=1, inplace=True)
    return train, test
