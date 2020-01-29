from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn import ensemble
from os.path import exists
import data_loader
import numpy as np
import pandas


#
def metrics(pred_val, true_val):
    # TPrate := recall
    TPrate = FPrate = precision = []
    tn = fn = fp = tp = 0
    best_acc = 0
    thr = 0
    n = len(true_val)
    for i in np.linspace(0, 1, 20):
        cm = confusion_matrix(true_val,  pred_val >= i)
        TPrate.append(cm[1][1] / (cm[1][1] + cm[0][1] + 1))
        FPrate.append(cm[1][0] / (cm[0][0] + cm[1][0] + 1))
        print("Threshold ", i)
        print(cm, "\n")
        precision.append(cm[1][1] / (cm[1][1] + cm[1][0]))
        accuracy = (cm[0][0] + cm[1][1]) / n
        if (accuracy > best_acc):
            tn = cm[0][0]
            fn = cm[0][1]
            fp = cm[1][0]
            tp = cm[1][1]
            thr = i
            best_acc = accuracy

    print("True Negative:  ", tn)
    print("True Positive:  ", tp)
    print("False Negative: ", fn)
    print("False Positive: ", fp)
    print("Best accuracy ", best_acc, " with threshold ", thr)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title('ROC Curve')
    plt.plot(np.array(FPrate), np.array(TPrate), 'r-')
    plt.subplot(2, 1, 2)
    plt.title('Precision-Recall Curve')
    plt.plot(TPrate, precision, 'b-')
    plt.show()


# Simple perceptron
class NeuralNetwork:
    # Creation neural network with 9 hidden layers
    def __init__(self, attack_type="Bot"):
        self.attack_type = attack_type
        self.download = False

        self.classifier = keras.Sequential()
        # Input layer
        self.classifier.add(keras.layers.Dense(30, activation=keras.activations.relu))
        # Hidden layers
        for i in range(0, 30):
            self.classifier.add(keras.layers.Dense(50, activation=keras.activations.relu))
            self.classifier.add(keras.layers.Dropout(0.1))
        # Output layer
        self.classifier.add(keras.layers.Dense(1, activation=keras.activations.sigmoid))
        self.classifier.compile(optimizer=keras.optimizers.SGD(0.1), loss='mse')

        # Try to find trained model
        self.model_path = "./models/" + self.attack_type + "/NeuralNetwork.h5"
        if (exists(self.model_path)):
            print("Load existing model...\n")
            self.classifier = keras.models.load_model(self.model_path)

    def __load(self):
        if not self.download:
            self.train_set, self.test_set = data_loader.load_set(self.attack_type)
            self.download = True

    def train(self):
        self.__load()
        print("Start to train neural network...")
        history = self.classifier.fit(self.train_set[0], self.train_set[1], batch_size=100, epochs=3, validation_split=0.15)
        self.classifier.save(self.model_path)
        print("Training is completed.\n")

    def test(self):
        self.__load()
        print("Start testing...\n")
        #self.classifier.evaluate(self.test_set[0], self.test_set[1])
        metrics(self.classifier.predict(self.test_set[0]), self.test_set[1])



    # Make plots of typical benign/anomaly packet by x and y numbers of features
    def typical_class(self, x, y):
        self.__load()
        label = np.reshape(np.array(self.test_set[1]), (len(self.test_set[1]), 1))
        df = pandas.DataFrame(data=np.hstack((self.test_set[0], label)))
        df.plot.scatter(x=x, y=y, c=76, colormap='viridis')
        plt.show()


# LSTM neural network
class LSTM:
    def __init__(self, attack_type="Bot"):
        self.attack_type = attack_type
        self.download = False

        self.classifier = keras.Sequential()
        self.classifier.add(keras.layers.LSTM(128, return_sequences=True))
        self.classifier.add(keras.layers.Dropout(0.25))
        self.classifier.add(keras.layers.LSTM(128, return_sequences=True))
        self.classifier.add(keras.layers.LSTM(128, return_sequences=False))
        self.classifier.add(keras.layers.Dropout(0.2))
        self.classifier.add(keras.layers.Dense(1, activation=keras.activations.sigmoid))
        self.classifier.compile(optimizer=keras.optimizers.SGD(0.1), loss='mean_absolute_error', metrics=['accuracy'])

        # Try to find trained model
        self.model_path = "./models/" + self.attack_type + "/LSTM.h5"
        if (exists(self.model_path)):
            print("Load existing model...\n")
            self.classifier = keras.models.load_model(self.model_path)

    def __load(self):
        if not self.download:
            self.train_set, self.test_set = data_loader.load_seq(self.attack_type)
            self.download = True

    def train(self):
        self.__load()
        print("Start to train LSTM-network...")
        history = self.classifier.fit(self.train_set[0], self.train_set[1], batch_size=50, validation_split=0.15, epochs=3)
        self.classifier.save(self.model_path)
        print("Training is completed.\n")
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()

    def test(self):
        self.__load()
        print("Start testing...\n")
        self.classifier.evaluate(self.test_set[0], self.test_set[1])
        metrics(self.classifier.predict(self.test_set[0]), self.test_set[1])

    def predict(self, x):
        return self.classifier.predict(x)


# Sklearn implementation of Random Forest classifier
class random_forest():
    def __init__(self):
        self.classifier = ensemble.RandomForestClassifier()


# TODO add k-NN nearest algorithm
# TODO try to make reinforcement learning
