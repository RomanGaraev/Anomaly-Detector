import matplotlib.pyplot as plt
from tensorflow import keras
import data_loader
import numpy as np
import pandas


# Simple perceptron
class NeuralNetwork:
    def __init__(self):
        # Creation neural network with 9 hidden layers
        self.classifier = keras.Sequential()
        # Input layer
        self.classifier.add(keras.layers.Dense(10, input_shape=(76,), activation=keras.activations.relu))
        # Hidden layers
        for i in range(0, 9):
            self.classifier.add(keras.layers.Dense(10, activation=keras.activations.relu))
        # Output layer
        self.classifier.add(keras.layers.Dense(2, activation=keras.activations.sigmoid))
        # ??
        self.classifier.compile(optimizer=keras.optimizers.Adam(0.01),
                                loss='mse',
                                metrics=['accuracy'])

    def Train(self):
        # Data loading did't fail
        if not (self.train_set == []):
            print("Start training neural network...")
            self.classifier.fit(self.train_set[0], self.train_set[1], epochs=3, batch_size=64,
                                validation_data=(self.valid_set[0], self.valid_set[1]))
            print("Training is  complete.\n")
        else:
            print("Data loading failed.\n")

    def Test(self):
        # Data loading did not fail
        if not (self.test_set == []):
            print("Start testing...\n")
            self.classifier.evaluate(self.test_set[0], self.test_set[1])

    # Return predicted class of sample x
    def Predict(self, x):
        return self.classifier.predict(x)

    def Save(self):
        a = 0

    def Load(self, attack_type):
        # Load the data
        self.train_set, self.valid_set, self.test_set = data_loader.LoadSet(attack_type)

    # Make plots of typical benign/anomaly packet
    # TODO did't work!
    def TypicalClass(self):
        a = np.reshape(np.array(self.test_set[1]), (len(self.test_set[1]), 1))
        df = pandas.DataFrame(data=np.hstack((self.test_set, a)))
        df.plot.scatter(x=1, y=76, c=76, colormap='viridis')
        plt.show()
        #print(df.groupby(76).mean())


#LSTM neural network
class LSTM:
    def __init__(self, attack_type):
        # Load the data
        self.train_set, self.valid_set, self.test_set = data_loader.LoadSet(attack_type)
        self.classifier = keras.Model



# TODO add k-NN nearest algorithm
# TODO add random forest or AdaBoost algorithms
# TODO try to make reinforcement learning
