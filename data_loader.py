from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas
import os


# Available data sets of attacks
attacks = ['Bot', 'Bot_enc', 'Brute force', 'DDoS', 'DoS', 'Infiltration', 'Web', 'Test', 'Test_enc']

# Names of the columns that should be dropped
drop_col = ['Timestamp', 'Flow Byts/s', 'Flow Pkts/s', 'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Flow duration']


# Find all .csv files in the folder "data\attack_type\"
def find_csv(attack_types=['Bot']):
    files = []
    for attack_type in attack_types:
        if not any(attack_type in name for name in attacks):
            raise ValueError("There is no type of attack you try to download in data set:" + attack_type + "\n")
        path = "data\\" + attack_type + "\\"
        for r, d, f in os.walk(path):
            print("Files in the folder", path + ":")
            for file in f:
                if '.csv' in file:
                    files.append(path + file)
                    print(file)
    return files


# Return pandas.DataFrame from .csv files
def load_frame(attack_types=['Bot']):
    files = find_csv(attack_types)
    tables = []
    print("\nStart loading...")
    # Download sets from the .csv tables
    for file in files:
        tables.append(pandas.read_csv(file, sep=","))
        print("File", file, "is added.")
    # Union of the DataFrames
    data_frame = pandas.concat(tables, ignore_index=True)
    print("Data set completely downloaded. Total length of samples:", len(data_frame), "\n")
    # Return without "bad" features, if they exist
    return data_frame.drop(set(drop_col) & set(data_frame.columns), axis=1)


# Return standardized train + validation and test sets as numpy arrays
def load_set(attack_types=['Bot'], train_size=0.85, shuffle=True):
    train, test = train_test_split(load_frame(attack_types), train_size=train_size, shuffle=shuffle)

    train_label = np.array(train['Label'].map({'Benign': 0, 'Bot': 1, 'No Label': 0}))
    test_label = np.array(test['Label'].map({'Benign': 0, 'Bot': 1, 'No Label': 0}))

    train = scale(train.drop('Label', axis=1))
    test = scale(test.drop('Label', axis=1))

    return [np.array(train), train_label.reshape(train_label.shape[0], 1)], \
           [np.array(test),  test_label.reshape(test_label.shape[0], 1)]


# Return divided by type (Benign or Malicious) standardized sets
def load_separate(attack_types=['Bot']):
    data_set = load_frame(attack_types)
    # Standardization
    features = pandas.DataFrame(data=scale(data_set.iloc[:, 0:len(data_set.columns) - 1]))
    separate = []
    separate.append(features.loc[data_set['Label'] == 'Benign'])
    # Different types of anomalies
    for i in range(0, len(attack_types)):
        separate.append(features.loc[data_set['Label'] == attack_types[i]])
    return separate


# Return train+validation and test sets as arrays of sequences for recurrent algorithms.
# Data set is sliced to sequences of length=step. Samples are from separated data sets
# noise - number of noise samples in the sequence
def load_seq(attack_types=['Bot'], step=10, train_size=0.85, noise=4):
    seq = []
    seq_lab = []
    # Benign=0 or malicious=1,2...
    label = 0
    Sets = load_separate(attack_types)

    print("Start creating sequences...\n")
    for set_ind in range(0, len(Sets)):
        # Going through set and adding the batches of samples to the sequence
        for i in range(0, len(Sets[set_ind]) - step, step - noise):
            # Pure sequence
            noise_seq = Sets[set_ind][i: i + step].values.tolist()
            # Add some noise to the pure sequence
            for j in range(0, noise):
                # We choose sample from random set and insert it to the random position
                rand_set = np.random.randint(0, len(Sets))
                rand_samp = Sets[rand_set].iloc[np.random.randint(0, len(Sets[rand_set]))]
                noise_seq.insert(np.random.randint(0, step), rand_samp)
            # len(noise_seq) = step + noise_step, we need only first step elements
            seq.append(np.array(noise_seq[0:step]))
            seq_lab.append([label])
        label += 1
    print("Total amount of sequences: ", len(seq), "\n")

    # Mixing rows
    index = np.random.permutation(np.arange(len(seq)))
    train_ind = index[: np.int32(train_size * len(seq))]
    test_ind = index[np.int32(train_size * len(seq)):]

    return [np.array([seq[i] for i in train_ind]), np.array([seq_lab[i] for i in train_ind])], \
           [np.array([seq[i] for i in test_ind]),  np.array([seq_lab[i] for i in test_ind])]


# Transform .csv file to the proper format
def transform(file_name, label='Benign'):
    file = load_frame(file_name)
    file['Label'] = label
    file.to_csv("data\\" + file_name[0] + "\\" + file_name[0] + ".csv", index=False)
    print(file_name, " was changed")


# Make plot of typical benign/anomaly packet by x and y numbers of features
def typical_class(x, y, attack_type="Bot"):
    df = load_frame([attack_type])
    df['Label'] = df['Label'].map({'Benign': 0, attack_type: 1})
    df.plot.scatter(x=x, y=y, c='Label', colormap='plasma')
    plt.show()