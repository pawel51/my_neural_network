from layer import Layer
import numpy as np


def relu(x):
    if x is not isinstance(x, (int, float)):
        print("x is not a number")
        return 0
    if x <= 0:
        return 0
    else:
        return x


def he(layers, index):
    if index == 0:
        size_l_prev = 1
        size_l = layers[0].get_node_list()
    else:
        size_l_prev = len(layers[index - 1].get_node_list())
        size_l = len(layers[index].get_node_list())

    return np.random.randn(size_l, size_l_prev) * np.sqrt(1 / size_l_prev)



