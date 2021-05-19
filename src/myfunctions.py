import numpy as np
import math as m

def L2(y, yhat, derivative=0):
    if derivative == 1:
        return L2_der(y, yhat)
    return m.pow(yhat-y, 2)

def L2_der(y, yhat):
    return 2*(yhat-y)

def BCE(y, yhat, derivative=0):
    if derivative == 1:
        return BCE_der(y, yhat)
    return y*m.log(yhat) + (1-y)*m.log(1-yhat)

def BCE_der(y, yhat):
    return (yhat-y)/(yhat*(1-yhat))

def relu(x, derivative=0):
    if derivative == 1:
        if x<0:
            return 0
        else:
            return 1

    if x <= 0:
        return 0
    else:
        return x

def relu_der(x):
    if x <= 0:
        return 0
    else:
        return 1

def sigmoid(x, derivative=0):
    if derivative == 1:
        return sigmoid_der(x)
    return 1/(1+m.exp(-x))

def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

def tanh(x, derivative=0):
    if derivative == 1:
        return tanh_der(x)
    return m.tanh(x)

def tanh_der(x):
    return 1-m.pow(tanh(x),2)

def he(layers, index):
    if index == 0:
        return
    else:
        size_l_prev = len(layers[index - 1].get_node_list())
        size_l = len(layers[index].get_node_list())

    return np.random.randn(size_l, size_l_prev) * np.sqrt(1 / size_l_prev)

def xavier(layers, index):
    if index == 0:
        return
    else:
        size_l_prev = len(layers[index - 1].get_node_list())
        size_l = len(layers[index].get_node_list())

    return np.random.randn(size_l,size_l_prev)*np.sqrt(2/(size_l_prev+size_l))



