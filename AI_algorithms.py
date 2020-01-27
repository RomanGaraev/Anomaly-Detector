import matplotlib.pyplot as plt
from tensorflow import keras
import data_loader
import numpy as np
import pandas
import os


# Simple perceptron
class NeuralNetwork:
    # Creation neural network with 9 hidden layers
    def __init__(self, attack_type="Bot"):
        self.attack_type = attack_type
        self.train_set, self.valid_set, self.test_set = data_loader.LoadSet(attack_type)

        self.classifier = keras.Sequential()
        # Input layer
        self.classifier.add(keras.layers.Dense(10, input_shape=(len(self.train_set[0].columns), ), activation=keras.activations.relu))
        # Hidden layers
        for i in range(0, 9):
            self.classifier.add(keras.layers.Dense(10, activation=keras.activations.relu))
        # Output layer
        self.classifier.add(keras.layers.Dense(2, activation=keras.activations.sigmoid))
        self.classifier.compile(optimizer=keras.optimizers.Adam(0.01), loss='mse', metrics=['accuracy'])

    def Train(self):
        # Try to find trained model
        self.model_path = "./models/" + self.attack_type + "/nn_weights.h5"
        if (os.path.exists(self.model_path)):
            print("Load existing model...\n")
            self.classifier.load_weights(self.model_path)
        else:
            # Training, if there's no saved weights found
            print("Start to train neural network...")
            self.classifier.fit(self.train_set[0], self.train_set[1], epochs=3, validation_data=self.valid_set)
            self.classifier.save_weights(self.model_path, save_format='h5')
            print("Training is completed.\n")

    def Test(self):
        print("Start testing...\n")
        self.classifier.evaluate(self.test_set[0], self.test_set[1])

    def Predict(self, x):
        return self.classifier.predict(x)

    # Make plots of typical benign/anomaly packet by x and y numbers of features
    def TypicalClass(self, x, y):
        label = np.reshape(np.array(self.test_set[1]), (len(self.test_set[1]), 1))
        df = pandas.DataFrame(data=np.hstack((self.test_set[0], label)))
        df.plot.scatter(x=x, y=y, c=76, colormap='viridis')
        plt.show()
        #print(df.groupby(76).mean())


# LSTM neural network
class LSTM:
    def __init__(self, attack_type="Bot"):
        self.attack_type = attack_type
        self.train_set, self.valid_set, self.test_set = data_loader.LoadSeq(attack_type)

        self.classifier = keras.Sequential()
        self.classifier.add(keras.layers.LSTM(128, return_sequences=True))
        self.classifier.add(keras.layers.LSTM(128, return_sequences=True))
        self.classifier.add(keras.layers.LSTM(128, return_sequences=False))
        self.classifier.add(keras.layers.Dropout(0.3))
        self.classifier.add(keras.layers.Dense(2, activation=keras.activations.sigmoid))
        self.classifier.compile(optimizer=keras.optimizers.Adam(0.2), loss='mse', metrics=['accuracy'])

    def Train(self):
        # Try to find trained model
        self.model_path = "./models/" + self.attack_type + "/LSTM_weights.h5"
        if (os.path.exists(self.model_path)):
            print("Load existing model...\n")
            self.classifier.load_weights(self.model_path)
        else:
            # Training, if there's no saved weights found
            print("Start to train LSTM-network...")

            print(self.train_set[0].shape, self.train_set[1].shape)
            self.classifier.fit(self.train_set[0], self.train_set[1], batch_size=30, validation_data=self.valid_set, epochs=3)
            self.classifier.save_weights(self.model_path, save_format='h5')
            print("Training is completed.\n")

    def Test(self):
        print("Start testing...\n")
        self.classifier.evaluate(self.test_set[0], self.test_set[1])

    def Predict(self, x):
        return self.classifier.predict(x)


# TODO add k-NN nearest algorithm
# TODO add random forest or AdaBoost algorithms
# TODO try to make reinforcement learning
