from utils import *
import numpy as np
from tqdm import tqdm

class Dense:
    
    def __init__(
        self, 
        in_features : int,
        out_features : int,
        shape : int, 
        activation = None,
    ):
        self.weights = np.random.rand(in_features, out_features)*2-1
        self.biases = np.random.rand(out_features, shape)
        
        if activation=='relu':
            self.act = relu
            self.actback = relu_back
        elif activation=='sigmoid':
            self.act = sigmoid
            self.actback = sigmoid_back
        elif activation=='softmax':
            self.act = softmax
            self.actback = softmax_back
        else:
            self.act = identity
            self.actback = identity_back
            
        self.batch_counter = 0
        
        self.dw = np.zeros(self.weights.shape)
        self.db = np.zeros(self.biases.shape)
            
    def forward(
        self, 
        x : np.ndarray,
    ):
        self.z = np.dot(self.weights.T,x)+self.biases
        self.a = self.act(self.z)
        return self.a
    
    def backprop(
        self, 
        a_prev_layer : np.ndarray, 
        da_this_layer : np.ndarray,
    ):
        self.batch_counter += 1
        da_by_dz = da_this_layer * self.actback(self.z)
        dz_by_db = np.ones(self.biases.shape)
        dz_by_dw = a_prev_layer
        self.db += dz_by_db * da_by_dz
        self.dw += np.dot(dz_by_dw, da_by_dz.T)
        da_prev_layer = np.dot(self.weights, da_by_dz)
        return da_prev_layer
    
    def update(self, learning_rate = 1e-3):
        if self.batch_counter > 0:

            self.dw = np.clip(self.dw / self.batch_counter, -1e5, 1e5)
            self.db = np.clip(self.db / self.batch_counter, -1e5, 1e5)

            self.weights = self.weights - self.dw * learning_rate 
            self.biases = self.biases - self.db * learning_rate
            self.batch_counter = 0

            self.dw = np.zeros(self.weights.shape)
            self.db = np.zeros(self.biases.shape)

class NeuralNet:
    
    def __init__(
        self, 
        dense_layers : list,
        cost_f,
        cost_f_back,
        learning_rate = 1e-4,
    ):
        self.layers = dense_layers
        self.cost_f = cost_f
        self.cost_f_back = cost_f_back
        self.learning_rate=learning_rate
        
    def all_forward(
        self,
        x : np.ndarray,
    ):
        self.input_value = x
        for layer in self.layers:
            x = layer.forward(x)
        return x
        
    def all_backprop(
        self,
        y_pred : np.ndarray,
        y_true : np.ndarray,
    ):
        da_curr = self.cost_f_back(y_pred, y_true)
        for i in range(len(self.layers)):
            idx = len(self.layers)-i-1
            if idx == 0:
                da_curr = self.layers[idx].backprop(
                    self.input_value,
                    da_curr,
                )
            else:
                da_curr = self.layers[idx].backprop(
                    self.layers[idx-1].a,
                    da_curr,
                )
                
    def all_update(self):
        for layer in self.layers:
            layer.update(learning_rate = self.learning_rate)
    
    def train_on_dataset(
        self,
        batch : np.ndarray,
        annotations : np.ndarray,
        batch_size = 100,
    ):
        cost_counter = []
        batch_counter = 0
        pbar = tqdm(zip(batch, annotations))
        for item, y_true in pbar:
            y_pred = self.all_forward(item)
            cost = self.cost_f(y_pred, y_true)
            self.all_backprop(y_pred, y_true)
            cost_counter.append(cost)
            batch_counter += 1
            if batch_counter == batch_size: 
                self.all_update()
                batch_counter=0
                pbar.set_description(f'cost = {round(np.average(cost_counter),2)}')
        cost_counter = np.average(cost_counter)        
        self.all_update()  
        
        return cost_counter
    
    def predict_on_batch(
        self,
        batch : np.ndarray,
    ):
        out = []
        for item in tqdm(batch):
            y_pred = self.all_forward(item)
            out.append(y_pred)
        return np.array(out)