import numpy as np
from tensorflow import keras # only imported to read MNIST
import matplotlib.pyplot as plt
from nn import *
from utils import *

epochs = 20

def onehot(y :int, n_class=10):
        out = []
        for i in range(n_class):
            if i==y:
                out.append(1)
            else:
                out.append(0)
        return np.array(out).reshape(-1,1)

if (__name__=='__main__'):

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    train_size = 60000
    test_size = 10000

    X_train = np.array([x_train[i].reshape(-1,1) for i in range(train_size)])/255
    y_train = np.array([onehot(y_train[i]) for i in range(train_size)])
    X_test = np.array([x_test[i].reshape(-1,1) for i in range(test_size)])/255
    y_test = np.array([onehot(y_test[i]) for i in range(test_size)])

    dense_layers = [
        Dense(784,69,1,'relu'),
        Dense(69,10,1,'softmax')
    ]

    nn = NeuralNet(dense_layers, BCE_cost_f, BCE_cost_f_back, learning_rate = 1e-5)
    cost_log = []
    for i in range(epochs):   
        cost = nn.train_on_dataset(X_train, y_train, batch_size=100)
        cost_log.append(cost)
        
    plt.plot(cost_log)
    plt.show()

    y_pred = nn.predict_on_batch(X_test)
    acc = np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)
    acc = np.sum(acc)/len(acc)
    print('Accuracy = ',round(acc*100,2))
