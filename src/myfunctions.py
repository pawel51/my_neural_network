import numpy as np
import math as m

def relu(x):
    if x <= 0:
        return 0
    else:
        return x

def relu_der(x):
    if x <= 0:
        return 0
    else:
        return 1

def sigmoid(x):
    return 1/(1+m.exp(-x))

def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

def tanh(x):
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



