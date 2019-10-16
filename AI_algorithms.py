import data_loader
from sklearn import neural_network

# Simple perceptron
class NeuralNetwork:
    def __init__(self, attack_type):
        self.train_set, self.valid_set, self.test_set, \
        self.train_label, self.valid_label, self.test_label = data_loader.LoadSet(attack_type)
        self.classifier = neural_network.MLPClassifier(hidden_layer_sizes=(9, 4), alpha=0.00001)

    def Train(self):
        print("Start training neural network...")
        self.classifier.fit(self.train_set, self.train_label)
        print("Training completed")

    def Test(self):
        print("Start testing...")
        print("Result:", self.classifier.score(self.test_set, self.test_label))

    def Predict(self, x):
        return self.classifier.predict(x)
