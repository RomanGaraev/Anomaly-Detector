from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import numpy as np
import pandas
import boto3
import os


# Available data sets of attacks
attacks = ['Bot', 'Bot_enc', 'Brute force', 'DDoS', 'DoS', 'Infiltration', 'Web', 'Test']

# The names of the columns that should be dropped
drop_col = ['Timestamp', 'Flow Byts/s', 'Flow Pkts/s', 'Flow ID', 'Src IP', 'Dst IP', 'Flow duration']


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
    # Download sets from the .csv tables
    for file in files:
        tables.append(pandas.read_csv(file, sep=","))
        print("File", file, "is added.")
    # Union of the DataFrames
    data_frame = pandas.concat(tables, ignore_index=True)
    # Drop "bad" features, if they exist
    data_frame.drop(set(drop_col) & set(data_frame.columns), axis=1)
    print("Data set completely downloaded. Total length of samples:", len(data_frame), "\n")
    return data_frame


# Return standardized train + validation and test sets as numpy arrays
def load_set(attack_type='Bot', train_size=0.85):
    train, test = train_test_split(load_frame(attack_type), train_size=train_size)

    train_label = np.array(train['Label'].map({'Benign': 0, attack_type: 1}))
    test_label = np.array(test['Label'].map({'Benign': 0, attack_type: 1}))

    train = scale(train.drop('Label', axis=1))
    test = scale(test.drop('Label', axis=1))

    return [train, train_label.reshape(train_label.shape[0], 1)], \
           [test,  test_label.reshape(test_label.shape[0], 1)]


# Return divided by type (Benign or Malicious) standardized sets
def load_separate(attack_type='Bot'):
    data_set = load_frame(attack_type)
    # Standardization
    features = pandas.DataFrame(data=scale(data_set.iloc[:, 0:len(data_set.columns) - 1]))
    return features.loc[data_set['Label'] == 'Benign'], features.loc[data_set['Label'] == attack_type]


# Return train+validation and test sets as arrays of sequences for recurrent algorithms.
# Data set is sliced to sequences of random length. Samples are from separated data sets
def load_seq(attack_type='Bot', step=20, train_size=0.85):
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
    train_ind = index[: np.int32(train_size * len(seq))]
    test_ind = index[np.int32(train_size * len(seq)):]

    return [np.array([seq[i] for i in train_ind]), np.array([seq_lab[i] for i in train_ind])], \
           [np.array([seq[i] for i in test_ind]),  np.array([seq_lab[i] for i in test_ind])]


# Download data set from Amazon Web Services
def load_aws():
    s3 = boto3.client('s3')
    response = s3.list_buckets()

    # Output the bucket names
    print('Existing buckets:')
    for bucket in response['Buckets']:
        print(f' {bucket["Name"]}')
    #s3 = boto3.client('s3')
    #s3.create_bucket(Bucket=)
    #s3.download_file("cse-cic-ids2018", "Processed Traffic Data for ML Algorithms", "/test")
