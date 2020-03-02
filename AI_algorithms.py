import data_loader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import neighbors
from tensorflow import keras
from sklearn import ensemble
from os.path import exists
from time import clock
import numpy as np
import pickle
import pandas


# Common function for classifiers that print accuracy metrics
def metrics(prediction, true_val):
    # TPrate := recall
    TPrate = FPrate = precision = []
    tn = fn = fp = tp = 0
    best_acc = 0
    thr = 0
    n = len(true_val)
    # Going through thresholds
    for i in np.linspace(0, 1, 21):
        cm = confusion_matrix(true_val,  prediction >= i)
        print("Threshold ", i)
        print(cm, "\n")
        TPrate.append(cm[1][1] / (cm[1][1] + cm[1][0] + 1))
        FPrate.append(cm[0][1] / (cm[0][0] + cm[0][1] + 1))
        precision.append(cm[1][1] / (cm[1][1] + cm[0][1] + 1))
        accuracy = (cm[0][0] + cm[1][1]) / n
        if accuracy > best_acc:
            tn = cm[0][0]
            fn = cm[1][0]
            fp = cm[0][1]
            tp = cm[1][1]
            thr = i
            best_acc = accuracy

    print("Best accuracy   ", best_acc, " with threshold ", thr)
    print("True Negative:  ", tn)
    print("True Positive:  ", tp)
    print("False Negative: ", fn)
    print("False Positive: ", fp)
    print("Precision:      ", tp / (tp + fp))
    print("Recall:         ", tp / (tp + fn))
    print("F1 score:       ", 2 * tp / (2 * tp + fn + fp))

    fig = plt.figure(figsize=(8, 8))
    fig1 = fig.add_subplot(211)
    fig1.plot(FPrate, TPrate, 'go')
    fig1.set_xlabel('FPrate')
    fig1.set_ylabel('TPrate')
    fig1.set_title('ROC Curve')

    fig2 = fig.add_subplot(212)
    fig2.plot(TPrate, precision, 'rx')
    fig2.set_xlabel('Recall')
    fig2.set_ylabel('Precision')
    fig2.set_title('Precision-Recall Curve')
    fig.tight_layout(pad=3.0)
    plt.show()


# Make plots of typical benign/anomaly packet by x and y numbers of features
def typical_class(x, y, attack_type="Bot"):
    df = data_loader.load_frame(attack_type)
    df['Label'] = df['Label'].map({'Benign': 0, attack_type: 1})
    df.plot.scatter(x=x, y=y, c='Label', colormap='plasma')
    plt.show()


# Sklearn implementation of Random Forest classifier
class RandomForest:
    def __init__(self, attack_type="Bot"):
        self.__attack_type = attack_type
        self.__download = False
        # Try to find existing model
        self.__model_path = "./models/" + self.__attack_type + "/RandomForest.sav"
        if exists(self.__model_path):
            print("Load existing model...\n")
            self.__classifier = pickle.load(open(self.__model_path, 'rb'))
        else:
            # TODO add more parameters
            self.__classifier = ensemble.RandomForestClassifier(criterion="entropy", max_depth=15)

    def __load(self):
        if not self.__download:
            self.__train_set, self.__test_set = data_loader.load_set(self.__attack_type)
            self.__download = True

    def train(self):
        self.__load()
        print("Start to train random forest...")
        start = clock()
        self.__classifier.fit(self.__train_set[0], self.__train_set[1])
        end = clock()
        pickle.dump(self.__classifier, open(self.__model_path, 'wb'))
        print("Training is completed. Total time of training is", end - start, "sec.\n")

    def test(self):
        self.__load()
        print("Start testing...\n")
        self.__classifier.score(self.__test_set[0], self.__test_set[1])
        metrics(self.__classifier.predict(self.__test_set[0]), self.__test_set[1])

    def predict(self, x, threshold=0.85):
        return self.__classifier.predict(x) >= threshold


# TODO learn more about k-NN
# Sklearn implementation of K-nearest neighbors classifier
class KNN:
    def __init__(self, attack_type="Bot"):
        self.__attack_type = attack_type
        self.__download = False
        # Try to find existing model
        self.__model_path = "./models/" + self.__attack_type + "/KNN.sav"
        if exists(self.__model_path):
            print("Load existing model...\n")
            self.__classifier = pickle.load(open(self.__model_path, 'rb'))
        else:
            self.__classifier = neighbors.KNeighborsClassifier(n_neighbors=2)

    def __load(self):
        if not self.__download:
            self.__train_set, self.__test_set = data_loader.load_set(self.__attack_type)
            self.__download = True

    def train(self):
        self.__load()
        print("Start to train K-nearest neighbors...")
        start = clock()
        self.__classifier.fit(self.__train_set[0], self.__train_set[1])
        end = clock()
        pickle.dump(self.__classifier, open(self.__model_path, 'wb'))
        print("Training is completed. Total time of training is", end - start, "sec.\n")

    def test(self):
        self.__load()
        print("Start testing...\n")
        self.__classifier.score(self.__test_set[0], self.__test_set[1])
        metrics(self.__classifier.predict(self.__test_set[0]), self.__test_set[1])

    def predict(self, x, threshold=0.85):
        return self.__classifier.predict(x) >= threshold


# Simple perceptron
class NeuralNetwork:
    # Creation neural network with 9 hidden layers
    def __init__(self, attack_type="Bot"):
        self.__attack_type = attack_type
        self.__download = False
        # Try to find existing model
        self.__model_path = "./models/" + self.__attack_type + "/NeuralNetwork.h5"
        if exists(self.__model_path):
            print("Load existing model...\n")
            self.__classifier = keras.models.load_model(self.__model_path)
        else:
            self.__classifier = keras.Sequential()
            # Input layer
            self.__classifier.add(keras.layers.Dense(30, activation=keras.activations.relu))
            # Hidden layers
            for i in range(0, 15):
                self.__classifier.add(keras.layers.Dense(30, activation=keras.activations.relu))
                self.__classifier.add(keras.layers.Dropout(0.1))
            # Output layer
            self.__classifier.add(keras.layers.Dense(1, activation=keras.activations.sigmoid))
            # Learning configuration
            self.__classifier.compile(optimizer=keras.optimizers.SGD(0.1), loss='mse')

    def __load(self):
        if not self.__download:
            self.__train_set, self.__test_set = data_loader.load_set(self.__attack_type)
            self.__download = True

    def train(self):
        self.__load()
        print("Start to train neural network...")
        start = clock()
        self.__classifier.fit(self.__train_set[0], self.__train_set[1], batch_size=100, epochs=3, validation_split=0.15)
        end = clock()
        self.__classifier.save(self.__model_path)
        print("Training is completed. Total time of training is", end - start, "sec.\n")

    def test(self):
        self.__load()
        print("Start testing...\n")
        self.__classifier.evaluate(self.__test_set[0], self.__test_set[1])
        metrics(self.__classifier.predict(self.__test_set[0]), self.__test_set[1])

    def predict(self, x, threshold=0.85):
        return self.__classifier.predict(x) >= threshold


class Autoencoder:
    def __init__(self, attack_type="Bot"):
            self.__attack_type = attack_type
            self.__download = False
            # Default value of input dimension(turns to proper value in self.__load())
            self.__dim = 76
            # Try to find existing model
            self.__model_path = "./models/" + self.__attack_type + "/Autoencoder.h5"
            if exists(self.__model_path):
                print("Load existing model...\n")
                self.__autoenc = keras.models.load_model(self.__model_path)
                print(self.__autoenc.summary())
                # TODO
                self.__encoder = keras.Model(self.__autoenc.layers[0], self.__autoenc.layers[2](self.__autoenc.layers[1]))
            else:
                self.__encoding_dim = 10
                input_img = keras.Input(shape=(self.__dim, ))
                self.__encoded1 = keras.layers.Dense(self.__encoding_dim * 2, activation='relu')(input_img)
                self.__encoded = keras.layers.Dense(self.__encoding_dim, activation='relu')(self.__encoded1)
                self.__decoded = keras.layers.Dense(self.__dim, activation='sigmoid')(self.__encoded)
                self.__autoenc = keras.Model(input_img, self.__decoded)
                self.__encoder = keras.Model(input_img, self.__encoded)
                #self.__decoder = keras.Model(self.__encoded, self.__decoded)
                self.__autoenc.compile(optimizer='adadelta', loss='binary_crossentropy')

    def __load(self):
        if not self.__download:
            self.__train_set, self.__test_set = data_loader.load_set(self.__attack_type, train_size=0.999)
            self.__download = True
            # Input shape := first train sample's shape
            self.__dim = self.__train_set[0][0].shape[0]

    def train(self):
        self.__load()
        print("Start to train autoencoder...")
        start = clock()
        self.__autoenc.fit(self.__train_set[0], self.__train_set[0], batch_size=50, validation_split=0.15, epochs=3)
        end = clock()
        self.__autoenc.save(self.__model_path)
        print("Training is completed. Total time of training is", end - start, "sec.\n")

    def encode(self, x):
        return self.__encoder.predict(x)

    # Save encoded data set
    def save(self):
        separ = data_loader.load_separate(self.__attack_type)

        ben = pandas.DataFrame(self.encode(separ[0]))
        mal = pandas.DataFrame(self.encode(separ[1]))
        # Add labels

        ben['Label'] = 'Benign'
        mal['Label'] = self.__attack_type + '_enc'

        new_table = pandas.concat([ben, mal],ignore_index=True)
        new_table.to_csv('./data/' + self.__attack_type + '_enc/' + self.__attack_type + '_enc.csv', index=False)


# LSTM neural network
class LSTM:
    def __init__(self, attack_type="Bot"):
        self.__attack_type = attack_type
        self.__download = False
        # Try to find existing model
        self.__model_path = "./models/" + self.__attack_type + "/LSTM.h5"
        if exists(self.__model_path):
            print("Load existing model...\n")
            self.__classifier = keras.models.load_model(self.__model_path)
        else:
            self.__classifier = keras.Sequential()
            self.__classifier.add(keras.layers.LSTM(128, return_sequences=True))
            self.__classifier.add(keras.layers.Dropout(0.25))
            self.__classifier.add(keras.layers.LSTM(128, return_sequences=True))
            self.__classifier.add(keras.layers.LSTM(128, return_sequences=False))
            self.__classifier.add(keras.layers.Dropout(0.2))
            self.__classifier.add(keras.layers.Dense(1, activation=keras.activations.sigmoid))
            self.__classifier.compile(optimizer=keras.optimizers.SGD(0.1), loss='mean_absolute_error')

    def __load(self):
        if not self.__download:
            self.__train_set, self.__test_set = data_loader.load_seq(self.__attack_type)
            self.__download = True

    def train(self):
        self.__load()
        print("Start to train LSTM-network...")
        start = clock()
        self.__classifier.fit(self.__train_set[0], self.__train_set[1], batch_size=50, validation_split=0.15, epochs=3)
        end = clock()
        self.__classifier.save(self.__model_path)
        print("Training is completed. Total time of training is", end - start, "sec.\n")

    def test(self):
        self.__load()
        print("Start testing...\n")
        self.__classifier.evaluate(self.__test_set[0], self.__test_set[1])
        metrics(self.__classifier.predict(self.__test_set[0]), self.__test_set[1])

    def predict(self, x, threshold=0.85):
        return self.__classifier.predict(x) >= threshold


# TODO !!!!! the most important part, try to do better
# Gated recurrent unit (the faster modification of Long Short Term Memory) neural network
class GRU:
    def __init__(self, attack_type="Bot"):
        self.__attack_type = attack_type
        self.__download = False
        # Try to find existing model
        self.__model_path = "./models/" + self.__attack_type + "/GRU.h5"
        if exists(self.__model_path):
            print("Load existing model...\n")
            self.__classifier = keras.models.load_model(self.__model_path)
        else:
            self.__classifier = keras.Sequential()
            self.__classifier.add(keras.layers.GRU(32, return_sequences=True))
            self.__classifier.add(keras.layers.Dropout(0.15))
            self.__classifier.add(keras.layers.GRU(32, return_sequences=False))
            self.__classifier.add(keras.layers.Dense(1, activation=keras.activations.sigmoid))
            self.__classifier.compile(optimizer=keras.optimizers.SGD(0.1), loss='mean_absolute_error')

    def __load(self):
        if not self.__download:
            self.__train_set, self.__test_set = data_loader.load_seq(self.__attack_type, step=10)
            self.__download = True

    def train(self):
        self.__load()
        print("Start to train GRU-network...")
        start = clock()
        self.__classifier.fit(self.__train_set[0], self.__train_set[1], batch_size=50, validation_split=0.15, epochs=3)
        end = clock()
        self.__classifier.save(self.__model_path)
        print("Training is completed. Total time of training is", end - start, "sec.\n")

    def test(self):
        self.__load()
        print("Start testing...\n")
        self.__classifier.evaluate(self.__test_set[0], self.__test_set[1])
        metrics(self.__classifier.predict(self.__test_set[0]), self.__test_set[1])

    def predict(self, x, threshold=0.85):
        return self.__classifier.predict(x) >= threshold


# TODO add callbacks
# TODO try to make reinforcement learning
