import numpy as np

def relu(x):
    return np.maximum(x,0)

def relu_back(x):
    return (x>0).astype(float)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_back(x):
    sigm = sigmoid(x)
    return sigm * (1-sigm)

def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp, axis=0, keepdims=1)

def softmax_back(x):
    exp = np.exp(x)
    expsum = np.sum(exp, axis=0, keepdims=1)
    return - exp * (exp-expsum) / expsum**2
    
def identity(x):
    return x

def identity_back(x):
    return np.ones(x.shape)

def l2_cost_f(y_pred, y_true):
    return np.sum((y_true - y_pred)**2)

def l2_cost_f_back(y_pred, y_true):
    return 2 * (y_pred - y_true)

def BCE_cost_f(y_pred, y_true):
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1-eps)
    return np.sum(-(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred)))

def BCE_cost_f_back(y_pred, y_true):
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1-eps)
    return - np.divide(y_true,y_pred) + np.divide((1-y_true),(1-y_pred))

def CCE_cost_f(y_pred, y_true):
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1-eps)
    return np.sum(-(y_true * np.log(y_pred)))

def CCE_cost_f_back(y_pred, y_true):
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1-eps)
    return - np.divide(y_true,y_pred)