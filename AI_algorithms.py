import data_loader
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from os.path import exists
import datetime
import pandas


class LSTM:
    def __init__(self, attack_types=["Bot"]):

        self.__attack_types = attack_types
        self.__download = False
        # Try to find existing model
        self.__model_path = "./models/" + "".join(self.__attack_types) + "/LSTM.h5"
        if exists(self.__model_path):
            print("Load existing model...\n")
            self.__classifier = keras.models.load_model(self.__model_path)
        else:
            self.__classifier = keras.Sequential()
            self.__classifier.add(keras.layers.LSTM(64, return_sequences=True))
            self.__classifier.add(keras.layers.Dropout(0.2))
            self.__classifier.add(keras.layers.LSTM(64, return_sequences=False))
            self.__classifier.add(keras.layers.Dropout(0.15))
            self.__classifier.add(keras.layers.Dense(10, activation=keras.activations.sigmoid))
            self.__classifier.add(keras.layers.Dense(1, activation=keras.activations.sigmoid))
            self.__classifier.compile(optimizer=keras.optimizers.SGD(0.1), loss='mean_absolute_error')

    def __load(self, path=""):
        if not self.__download:
            if path == '':
                self.__train_set, self.__test_set = data_loader.load_seq(self.__attack_types)
            else:
                self.__train_set, self.__test_set = data_loader.load_seq(attack_types=[path], train_size=0.01, noise=0)
            self.__download = True

    def train(self):
        self.__load()
        print("Start to train LSTM-network...")
        start = datetime.datetime.now()
        self.__classifier.fit(self.__train_set[0], self.__train_set[1], batch_size=50, validation_split=0.15, epochs=3)
        end = datetime.datetime.now()
        self.__classifier.save(self.__model_path)
        print("Training is completed. Total time of training is", (end - start).seconds, "sec.\n")

    def test(self):
        self.__load()
        print("Start testing...\n")
        print("Confusion matrix:")
        print(confusion_matrix(self.__test_set[1], self.predict(self.__test_set[0])))
        print(classification_report(self.__test_set[1], self.__classifier.predict(self.__test_set[0]) >= 0.5,
                                    target_names=["Benign", "Anomaly"]))
        print("Test is over.")

    def speed_test(self, path=""):
        self.__load(path)
        benchmarks = []
        speed_set = self.__test_set[0][0: 100]
        for i in range(0, 100):
            start = datetime.datetime.now()
            test = self.predict(speed_set)
            end = datetime.datetime.now()
            benchmarks.append((end - start).microseconds / 100)
        benchmarks = sorted(benchmarks)
        print(benchmarks)
        print("Median:", benchmarks[int(len(benchmarks)/2)])
        print("Average:", sum(benchmarks)/len(benchmarks))
        print(self.__classifier.summary())

    def predict(self, x, threshold=0.85):
        return self.__classifier.predict(x) >= threshold


# Simple perceptron
class NeuralNetwork:
    # Creation neural network with 3 hidden layers
    def __init__(self, attack_types=["Bot"]):
        self.__attack_types = attack_types
        self.__download = False
        # Try to find existing model
        self.__model_path = "./models/" + "".join(self.__attack_types) + "/NeuralNetwork.h5"
        if exists(self.__model_path):
            print("Load existing model...\n")
            self.__classifier = keras.models.load_model(self.__model_path)
        else:
            self.__classifier = keras.Sequential()
            # Hidden layers
            for i in range(0, 3):
                self.__classifier.add(keras.layers.Dense(16, activation=keras.activations.relu))
            self.__classifier.add(keras.layers.Dropout(0.2))
            # Output layer
            self.__classifier.add(keras.layers.Dense(1, activation=keras.activations.sigmoid))
            # Learning configuration
            self.__classifier.compile(optimizer=keras.optimizers.SGD(0.15), loss='mse')

    def __load(self, path=''):
        if not self.__download:
            if path == '':
                self.__train_set, self.__test_set = data_loader.load_set(self.__attack_types)
            else:
                self.__train_set, self.__test_set = data_loader.load_set(attack_types=[path], train_size=0.01)
            self.__download = True

    def train(self):
        self.__load()
        print("Start to train neural network...")
        start = datetime.datetime.now()
        self.__classifier.fit(self.__train_set[0], self.__train_set[1], batch_size=100, epochs=3, validation_split=0.15)
        end = datetime.datetime.now()
        self.__classifier.save(self.__model_path)
        print("Training is completed. Total time of training is", (end - start).seconds, "sec.\n")

    # Path of different test set
    def test(self, path=""):
        self.__load(path)
        print("Start testing...\n")
        self.__classifier.evaluate(self.__test_set[0], self.__test_set[1])
        print("Confusion matrix:")
        print(confusion_matrix(self.__test_set[1], self.predict(self.__test_set[0])))
        print(classification_report(self.__test_set[1], self.__classifier.predict(self.__test_set[0]) >= 0.5,
                                    target_names=["Benign", "Anomaly"]))
        print("Test is over.")

    def speed_test(self, path=""):
        self.__load(path)
        benchmarks = []
        speed_set = self.__test_set[0][0: 100]
        for i in range(0, 1100):
            start = datetime.datetime.now()
            test = self.predict(speed_set)
            end = datetime.datetime.now()
            benchmarks.append((end - start).microseconds / 100)
        benchmarks = sorted(benchmarks)
        benchmarks = benchmarks[50:1050]
        print(benchmarks)
        print("Median:", benchmarks[int(len(benchmarks)/2)])
        print("Average:", sum(benchmarks)/len(benchmarks))

    def predict(self, x, threshold=0.5):
        return self.__classifier.predict(x) >= threshold


class Autoencoder:
    def __init__(self, attack_types=["Bot"]):
            self.__attack_types = attack_types
            self.__download = False
            # Default value of input dimension(turns to proper value in self.__load())
            self.__dim = 76
            # Try to find existing model
            self.__model_path = "./models/" + "".join(self.__attack_types) + "/Autoencoder.h5"
            if exists(self.__model_path):
                print("Load existing model...\n")
                self.__autoenc = keras.models.load_model(self.__model_path)
                # Divide the downloaded autoencoder to encoder and decoder
                self.__encoder = keras.Sequential()
                self.__encoder.add(self.__autoenc.layers[0])
                self.__encoder.add(self.__autoenc.layers[1])
                self.__encoder.add(self.__autoenc.layers[2])

            else:
                self.__encoding_dim = 8
                input_img = keras.Input(shape=(self.__dim, ))
                self.__encoded1 = keras.layers.Dense(self.__encoding_dim * 2, activation='relu')(input_img)
                self.__encoded = keras.layers.Dense(self.__encoding_dim, activation='linear')(self.__encoded1)
                self.__decoded1 = keras.layers.Dense(self.__encoding_dim * 3, activation='relu')(self.__encoded)
                self.__decoded1 = keras.layers.Dense(self.__encoding_dim * 3, activation='relu')(self.__decoded1)
                self.__decoded1 = keras.layers.Dense(self.__encoding_dim * 4, activation='relu')(self.__decoded1)
                self.__decoded = keras.layers.Dense(self.__dim, activation=keras.activations.linear)(self.__decoded1)
                self.__autoenc = keras.Model(input_img, self.__decoded)
                self.__encoder = keras.Model(input_img, self.__encoded)
                self.__autoenc.compile(optimizer=keras.optimizers.Adadelta(), loss=keras.metrics.mean_squared_error)

    def __load(self, path=''):
        if not self.__download:
            if path == '':
                self.__train_set, self.__test_set = data_loader.load_set(self.__attack_types, train_size=0.999)
            else:
                self.__train_set, self.__test_set = data_loader.load_set(attack_types=[path], train_size=0.999)
            self.__download = True
            # Input shape := first train sample's shape
            self.__dim = self.__train_set[0][0].shape[0]

    def train(self):
        self.__load()
        print("Start to train autoencoder...")
        start = datetime.datetime.now()
        self.__autoenc.fit(self.__train_set[0], self.__train_set[0], batch_size=50, validation_split=0.15, epochs=5)
        end = datetime.datetime.now()
        self.__autoenc.save(self.__model_path)
        print("Training is completed. Total time of training is", (end - start).seconds, "sec.\n")

    def encode(self, x):
        return self.__encoder.predict(x)

    def encode_decode(self, x):
        return self.__autoenc.predict(x)

    def speed_test(self, path=""):
        self.__load(path)
        benchmarks = []
        speed_set = self.__train_set[0][0: 100]
        for i in range(0, 1100):
            start = datetime.datetime.now()
            test = self.encode(speed_set)
            end = datetime.datetime.now()
            benchmarks.append((end - start).microseconds / 100)
        benchmarks = sorted(benchmarks)
        benchmarks = benchmarks[50:1150]
        print(benchmarks)
        print("Median:", benchmarks[int(len(benchmarks)/2)])
        print("Average:", sum(benchmarks)/len(benchmarks))

    # Save encoded data set
    def save(self, path=['']):
        if path == ['']:
            path = self.__attack_types
        separ = data_loader.load_separate(path)

        ben = pandas.DataFrame(self.encode(separ[0]))
        mal = pandas.DataFrame(self.encode(separ[1]))
        # Add labels
        ben['Label'] = 'Benign'
        mal['Label'] = self.__attack_types[0] + '_enc'

        new_table = pandas.concat([mal, ben], ignore_index=True)
        new_table.to_csv('./data/' + path[0] + '_enc/' + path[0] + '_enc.csv', index=False)


# Gated recurrent unit (the faster modification of Long Short Term Memory) neural network
class GRU:
    def __init__(self, attack_types=["Bot"]):
        self.__attack_types = attack_types
        self.__download = False
        # Try to find existing model
        self.__model_path = "./models/" + "".join(self.__attack_types) + "/GRU.h5"
        if exists(self.__model_path):
            print("Load existing model...\n")
            self.__classifier = keras.models.load_model(self.__model_path)
        else:
            self.__classifier = keras.Sequential()
            self.__classifier.add(keras.layers.GRU(64, activation=keras.activations.relu, return_sequences=True))
            self.__classifier.add(keras.layers.Dropout(0.2))
            self.__classifier.add(keras.layers.GRU(64, activation=keras.activations.relu, return_sequences=False))
            self.__classifier.add(keras.layers.Dropout(0.15))
            self.__classifier.add(keras.layers.Dense(10, activation=keras.activations.relu))
            self.__classifier.add(keras.layers.Dense(1, activation=keras.activations.sigmoid))
            self.__classifier.compile(optimizer=keras.optimizers.SGD(0.1), loss='mean_absolute_error')

    def __load(self, path=''):
        if not self.__download:
            if path == '':
                self.__train_set, self.__test_set = data_loader.load_seq(self.__attack_types)
            else:
                self.__train_set, self.__test_set = data_loader.load_seq(attack_types=[path], train_size=0.01, noise=0)
            self.__download = True

    def train(self):
        self.__load()
        print("Start to train GRU-network...")
        start = datetime.datetime.now()
        self.__classifier.fit(self.__train_set[0], self.__train_set[1], batch_size=50, validation_split=0.15, epochs=5)
        end = datetime.datetime.now()
        self.__classifier.save(self.__model_path)
        print("Training is completed. Total time of training is", (end - start).seconds, "sec.\n")

    # Path of different test set
    def test(self, path=""):
        self.__load(path)
        print("Start testing...\n")
        self.__classifier.evaluate(self.__test_set[0], self.__test_set[1])
        print("Confusion matrix:")
        print(confusion_matrix(self.__test_set[1], self.predict(self.__test_set[0]) >= 0.5))
        print(classification_report(self.__test_set[1], self.__classifier.predict(self.__test_set[0]) >= 0.5,
                                    target_names=["Benign", "Anomaly"]))

    def speed_test(self, path=""):
        self.__load(path)
        benchmarks = []
        speed_set = self.__test_set[0][0: 100]
        for i in range(0, 100):
            start = datetime.datetime.now()
            test = self.predict(speed_set)
            end = datetime.datetime.now()
            benchmarks.append((end - start).microseconds / 100)
        benchmarks = sorted(benchmarks)
        print(benchmarks)
        print("Median:", benchmarks[int(len(benchmarks)/2)])
        print("Average:", sum(benchmarks)/len(benchmarks))
        print(self.__classifier.summary())

    def predict(self, x, threshold=0.3):
        return self.__classifier.predict(x) >= threshold
