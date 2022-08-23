import numpy as np
from tensorflow import keras # only imported to read MNIST
import matplotlib.pyplot as plt
from nn import *
from utils import *

def onehot(y :int, n_class=10):
        out = []
        for i in range(n_class):
            if i==y:
                out.append(1)
            else:
                out.append(0)
        return np.array(out).reshape(-1,1)

#if (__name__=='main'):

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

batch_size = 1000

def onehot(y :int, n_class=10):
    out = []
    for i in range(n_class):
        if i==y:
            out.append(1)
        else:
            out.append(0)
    return np.array(out).reshape(-1,1)

batch_x = np.array([x_train[i].reshape(-1,1) for i in range(batch_size)])/255
batch_y = np.array([onehot(y_train[i]) for i in range(batch_size)])

dense_layers = [
    Dense(784,69,1,None),
    Dense(69,10,1,'softmax')
]

nn = NeuralNet(dense_layers, BCE_cost_f, BCE_cost_f_back, learning_rate = 1e-3)
cost_log = []
for i in range(100):   
    cost = nn.train_on_batch(batch_x, batch_y)
    cost_log.append(cost)
    print(cost)
    
plt.plot(cost_log)
plt.show()

y_pred = nn.predict_on_batch(batch_x)
acc = np.argmax(y_pred, axis=1) == np.argmax(batch_y, axis=1)
acc = np.sum(acc)/len(acc)
print('Accuracy = ',round(acc*100,2))
