import pandas
import numpy as np
from sklearn import preprocessing
import os

# Find all .csv files in the folder "data\attack_type\"
def FindCsv(attack_type):
    files = []
    path = "data\\" + attack_type + "\\"
    for r, d, f in os.walk(path):
        print("Founded files in the folder", path, ":")
        for file in f:
            if '.csv' in file:
                files.append(path + file)
                print(file)
    return files

# Return standardized train, validation and test sets
def LoadSet(attack_type):
    files = FindCsv(attack_type)
    tables = []
    print("\nLoading started...")
    # Download sets from the .csv tables without "bad" features
    for file in files:
        table = pandas.read_csv(file, sep=",").drop(['Timestamp', 'Flow Byts/s', 'Flow Pkts/s'], axis=1)
        tables.append(table)
        print("File", file, "is added.")

    # Union of the DataFrames
    data_set = pandas.concat(tables, ignore_index=True)
    features_len = len(data_set.columns) - 1
    print("Data set is completely loaded. Total length of samples:", len(data_set))

    # Mixing rows
    index = np.random.permutation(np.arange(len(data_set)))
    train_ind = index[: np.int32(0.70 * len(data_set))]
    valid_ind = index[np.int32(0.70 * len(data_set)) : np.int32(0.85 * len(data_set))]
    test_ind  = index[np.int32(0.85 * len(data_set)) :]

    # Standardization
    train_set = preprocessing.scale(data_set.iloc[train_ind, 0:features_len])
    valid_set = preprocessing.scale(data_set.iloc[valid_ind, 0:features_len])
    test_set  = preprocessing.scale(data_set.iloc[test_ind,  0:features_len])

    train_label = data_set.iloc[train_ind, features_len + 1]
    valid_label = data_set.iloc[valid_ind, features_len + 1]
    test_label  = data_set.iloc[test_ind,  features_len + 1]

    return train_set, valid_set, test_set, train_label, valid_label, test_label
