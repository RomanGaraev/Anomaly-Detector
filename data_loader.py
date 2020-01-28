from sklearn.preprocessing import scale
from sklearn import model_selection
import numpy as np
import pandas
import os


# Available data sets of attacks
attacks = ['Bot', 'Brute force', 'DDoS', 'DoS', 'Infiltration', 'Web']

# The names of the columns that should be dropped
drop_col = ['Timestamp', 'Flow Byts/s', 'Flow Pkts/s']


# Find all .csv files in the folder "data\attack_type\"
def find_csv(attack_type='Bot'):
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
def load_frame(attack_type='Bot'):
    files = find_csv(attack_type)
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


# Return standardized train + validation and test sets
def load_set(attack_type='Bot'):
    train, test = model_selection.train_test_split(load_frame(attack_type), train_size=0.85)

    train_label = train['Label'].map({'Benign': 0, attack_type: 1})
    test_label = test['Label'].map({'Benign': 0, attack_type: 1})

    train = pandas.DataFrame(scale(train.drop('Label', axis=1)))
    test = pandas.DataFrame(scale(test.drop('Label', axis=1)))

    return [train, train_label], [test, test_label]


# Return divided by type (Benign or Malicious) standardized sets
def load_separate(attack_type='Bot'):
    data_set = load_frame(attack_type)
    # Data set without labels
    features_len = len(data_set.columns) - 1
    # Standardization
    features = pandas.DataFrame(data=scale(data_set.iloc[:, 0:features_len]))
    return features.loc[data_set['Label'] == 'Benign'], features.loc[data_set['Label'] == attack_type]


# Return train+validation and test sets as arrays of sequences for recurrent algorithms.
# Data set is sliced to sequences of random length. Samples are from separated data sets
def load_seq(attack_type='Bot', step=20):
    seq = []
    seq_lab = []
    # Benign=0 or malicious=1
    label = 0

    print("Start creating sequences...\n")
    for Set in load_separate(attack_type):
        # Going through set and adding the batches of samples to the sequence
        for i in range(0, len(Set) - step, step):
            seq.append(np.array(Set[i: i + step]))
            seq_lab.append([label])
        label += 1
    print("Total amount of sequences: ", len(seq), "\n")

    # Mixing rows
    index = np.random.permutation(np.arange(len(seq)))
    train_ind = index[: np.int32(0.85 * len(seq))]
    test_ind = index[np.int32(0.85 * len(seq)):]

    return [np.array([seq[i] for i in train_ind]), np.array([seq_lab[i] for i in train_ind])], \
           [np.array([seq[i] for i in test_ind]),  np.array([seq_lab[i] for i in test_ind])]

