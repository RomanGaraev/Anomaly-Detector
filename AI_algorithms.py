import data_loader
import matplotlib.pyplot as plt
import pandas.plotting as pd
from sklearn import neural_network

# Визуализация данных
def GraphicAnalize(set):
    print("Start plotting...")
    data = set[['Dst Port', 'Fwd Seg Size Min', 'Protocol', 'Label', 'Fwd Pkt Len Max']]
    print(data.describe(include='all'))
    colors = {'Benign': 'green', 'Bot': 'red'}
    grr = pd.scatter_matrix(data, figsize=(10, 10), c=data['Label'].replace(colors),diagonal='kde', alpha=0.2)
    plt.show()

def NeuralNetwork():
    train_set, valid_set, test_set, train_label, valid_label, test_label = data_loader.LoadSet()
    classifier = neural_network.MLPClassifier(hidden_layer_sizes=(9, 4), alpha=0.00001)
    print("Start training neural network...")
    classifier.fit(train_set, train_label)
    print("Training completed")
    print("Start testing...")
    print("Result:", classifier.score(test_set,test_label))

