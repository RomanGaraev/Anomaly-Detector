from sklearn import preprocessing
import numpy as np
import pandas
import os


# Available data sets of attacks
attacks = ['Bot', 'Brute force', 'DDoS', 'DoS', 'Infiltration', 'Web']

# The names of the columns that should be dropped
drop_col = ['Timestamp', 'Flow Byts/s', 'Flow Pkts/s']


# Find all .csv files in the folder "data\attack_type\"
def FindCsv(attack_type='Bot'):
    if not any(attack_type in name for name in attacks):
        raise ValueError("There is no type of attack you try to download in data set!\n")
    files = []
    path = "data\\" + attack_type + "\\"
    for r, d, f in os.walk(path):
        print("Files in the folder", path + ":")
        for file in f:
            if '.csv' in file:
                files.append(path + file)
                print(file)
    return files


# Return pandas.DataFrame from .csv files
def LoadFrame(attack_type='Bot'):
    files = FindCsv(attack_type)
    tables = []
    print("\nStart loading...")
    # Download sets from the .csv tables without "bad" features
    for file in files:
        table = pandas.read_csv(file, sep=",").drop(drop_col, axis=1)
        tables.append(table)
        print("File", file, "is added.")
    # Union of the DataFrames
    data_set = pandas.concat(tables, ignore_index=True)
    print("Data set completely downloaded. Total length of samples:", len(data_set), "\n")
    return data_set


# Return standardized train, validation and test sets
def LoadSet(attack_type='Bot'):
    data_set = LoadFrame(attack_type)
    features_len = len(data_set.columns) - 1

    # Mixing rows
    index = np.random.permutation(np.arange(len(data_set)))
    train_ind = index[: np.int32(0.70 * len(data_set))]
    valid_ind = index[np.int32(0.70 * len(data_set)) : np.int32(0.85 * len(data_set))]
    test_ind  = index[np.int32(0.85 * len(data_set)) :]

    # Standardization
    train_features = pandas.DataFrame(data=preprocessing.scale(data_set.iloc[train_ind, 0:features_len]))
    valid_features = pandas.DataFrame(data=preprocessing.scale(data_set.iloc[valid_ind, 0:features_len]))
    test_features  = pandas.DataFrame(data=preprocessing.scale(data_set.iloc[test_ind,  0:features_len]))

    # Remapping 'Label' column to 0 and 1
    train_label = data_set.iloc[train_ind, features_len].map({'Benign': 0, attack_type: 1})
    valid_label = data_set.iloc[valid_ind, features_len].map({'Benign': 0, attack_type: 1})
    test_label  = data_set.iloc[test_ind,  features_len].map({'Benign': 0, attack_type: 1})

    # Types of return values are DataFrame and Series respectively
    return [train_features, train_label], [valid_features, valid_label], [test_features, test_label]


# Return divided by type (Benign or Malicious) standardized set
def LoadSeparate(attack_type='Bot'):
    data_set = LoadFrame(attack_type)
    # Data set without labels
    features_len = len(data_set.columns) - 1
    # Standardization
    features = pandas.DataFrame(data=preprocessing.scale(data_set.iloc[:, 0:features_len]))
    return features.loc[data_set['Label'] == 'Benign'],\
           features.loc[data_set['Label'] == attack_type]


# Return train, validation and test sets as arrays of sequences for recurrent algorithms.
# Data set is sliced to sequences of random length. Samples are from separated data sets
def LoadSeq(attack_type='Bot', min_len=10, max_len=20):
    # Download benign and malicious sets
    sets = LoadSeparate(attack_type)
    seq = []
    label = []
    # Benign=0 or malicious=1
    l = 0
    print("Start creating sequences...\n")
    for set in sets:
        i = 0
        data_len = len(set)
        # Going through set and adding sequences of samples to array
        while (i < data_len):
            size = 20 # size = min(np.random.randint(min_len, max_len), data_len - i)
            if (size + i < data_len):
             seq.append(np.array(set[i: i + size]))
             label.append([l])
            i += size
        l += 1

    print("Total amount of sequences: ", len(seq), "\n")

    # Mixing rows
    index = np.random.permutation(np.arange(len(seq)))
    train_ind = index[: np.int32(0.70 * len(seq))]
    valid_ind = index[np.int32(0.70 * len(seq)): np.int32(0.85 * len(seq))]
    test_ind  = index[np.int32(0.85 * len(seq)):]

    return [np.array([seq[i] for i in train_ind]), np.array([label[i] for i in train_ind])], \
           [np.array([seq[i] for i in valid_ind]), np.array([label[i] for i in valid_ind])], \
           [np.array([seq[i] for i in test_ind]),  np.array([label[i] for i in test_ind])]

