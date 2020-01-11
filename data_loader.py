from sklearn import preprocessing
import numpy as np
import pandas
import os


# Available data sets of attacks
Attacks = ['Bot', 'Brute force', 'DDoS', 'DoS', 'Infiltration', 'Web']

# The name of the columns that should be dropped
drop_col = ['Timestamp', 'Flow Byts/s', 'Flow Pkts/s']


# Find all .csv files in the folder "data\attack_type\"
def FindCsv(attack_type):
    if not any(attack_type in name for name in Attacks):
        return []
    files = []
    path = "data\\" + attack_type + "\\"
    for r, d, f in os.walk(path):
        print("Founded files in the folder", path + ":")
        for file in f:
            if '.csv' in file:
                files.append(path + file)
                print(file)
    return files

# Return standardized train, validation and test sets
def LoadSet(attack_type):
    files = FindCsv(attack_type)
    if(files == []):
        print("There is no type of attack you try to download in data set!\n")
        return [], [], []

    tables = []
    print("\nLoading started...")
    # Download sets from the .csv tables without "bad" features
    for file in files:
        table = pandas.read_csv(file, sep=",").drop(drop_col, axis=1)
        tables.append(table)
        print("File", file, "is added.")

    # Union of the DataFrames
    data_set = pandas.concat(tables, ignore_index=True)
    features_len = len(data_set.columns) - 1
    print("Data set is completely loaded. Total length of samples:", len(data_set), "\n")

    # Mixing rows
    index = np.random.permutation(np.arange(len(data_set)))
    train_ind = index[: np.int32(0.70 * len(data_set))]
    valid_ind = index[np.int32(0.70 * len(data_set)) : np.int32(0.85 * len(data_set))]
    test_ind  = index[np.int32(0.85 * len(data_set)) :]

    # Standardization
    train_feat = preprocessing.scale(data_set.iloc[train_ind, 0:features_len])
    valid_feat = preprocessing.scale(data_set.iloc[valid_ind, 0:features_len])
    test_feat  = preprocessing.scale(data_set.iloc[test_ind,  0:features_len])

    train_label = data_set.iloc[train_ind, features_len].map({'Benign': 0, attack_type: 1})
    valid_label = data_set.iloc[valid_ind, features_len].map({'Benign': 0, attack_type: 1})
    test_label  = data_set.iloc[test_ind,  features_len].map({'Benign': 0, attack_type: 1})

    return [train_feat, train_label], [valid_feat, valid_label], [test_feat, test_label]
