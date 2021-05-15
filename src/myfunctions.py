import numpy as np


def relu(x):
    if x <= 0:
        return 0
    else:
        return x

def sigmoid(x):
    pass



def he(layers, index):
    if index == 0:
        return
    else:
        size_l_prev = len(layers[index - 1].get_node_list())
        size_l = len(layers[index].get_node_list())

    return np.random.randn(size_l, size_l_prev) * np.sqrt(1 / size_l_prev)



