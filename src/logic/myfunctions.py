import math as m
import numpy as np


##
##  <-- Losses -->
##

def log(yhat, y, outputs, derivative=0):
    if yhat == 0:
        yhat += 0.001

    if derivative == 1:
        return -1 / yhat
    if y == 1:
        return -np.log(yhat)
    else:
        if yhat > 1:
            yhat = 0.999
        return -1 * np.log(1 - yhat)


def CE_v2_der(yhat, y, sum):
    if y == 1:
        return np.exp(yhat) / sum - 1
    elif y == 0:
        return -1 * np.exp(yhat) / sum


def CE_v2(yhat, y, outputs, derivative=0):
    sum = np.sum(np.exp(outputs)) + 0.00000001
    if derivative == 1:
        return CE_v2_der(yhat, y, sum)
    if y == 1:
        return -1 * np.log(np.exp(yhat) / sum)
    if y == 0:
        return -1 * np.log(1 - np.exp(yhat) / sum)


def CEL(yhat, y, outputs, derivative=0):
    if derivative == 1:
        return CEL_der(yhat, y, outputs)
    return -1 * np.log((np.exp(yhat) / np.sum(np.exp(outputs))) + 0.0001)


def CEL_der(yhat, y, outputs):
    return np.exp(yhat) / np.sum(np.exp(outputs)) - 1


# def hinge_der(y, yhat, outputs):
#     return ????
#
# def hinge(y, yhat, outputs, derivative=0):
#     if derivative == 1:
#         return hinge_der(y, yhat, outputs)
#     maxes = []
#     for out in outputs:
#         maximum = np.max(np.append(out - np.max(np.array(outputs) + 1, 0)))
#         maxes.append(maximum)
#     return np.sum(np.array(maxes))

def L2(y, yhat, outputs, derivative=0):
    if derivative == 1:
        return L2_der(y, yhat)
    return np.power(yhat - y, 2)


def L2_der(y, yhat):
    return 2 * (yhat - y)


def L1(y, yhat, outputs, derivative=0):
    if derivative == 1:
        return L1_der(y, yhat)
    return np.abs(yhat - y)


def L1_der(y, yhat):
    if yhat > y:
        return 1
    elif yhat == y:
        return 0
    else:
        return -1


def BCE(y, yhat, outputs, derivative=0):
    if derivative == 1:
        return BCE_der(y, yhat)
    if y == 0:
        if yhat >= 1:
            return 21
        return (1 - y) * np.log(1 - yhat)
    elif y == 1:
        if yhat <= 0:
            return 21
        return y * np.log(yhat)


def BCE_der(y, yhat):
    if yhat == 0:
        yhat = 0.00000001
    if yhat == 1:
        yhat = 1.00000001
    return (yhat - y) / (yhat * (1 - yhat))


##
##  <-- Activations -->
##




def relu(x, derivative=0):
    if derivative == 1:
        if x < 0:
            return 0
        else:
            return 1

    if x <= 0:
        return 0
    elif x>=10:
        return 10
    else:
        return x


def relu_der(x):
    if x <= 0:
        return 0
    else:
        return 1


def leaky_relu(x, derivative=0):
    if derivative == 1:
        if x < 0:
            return x
        else:
            return 1

    if x <= 0:
        return x * 0.01
    elif x > 10:
        return 10
    else:
        return x


def sigmoid(x, derivative=0):
    if derivative == 1:
        return sigmoid_der(x)
    try:
        return 1 / (1 + m.exp(-x))
    except OverflowError:
        return 0


def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x, derivative=0):
    if derivative == 1:
        return tanh_der(x)
    return m.tanh(x)


def tanh_der(x):
    return 1 - m.pow(m.tanh(x), 2)


def he(layers, index):
    if index == 0:
        return
    else:
        size_l_prev = len(layers[index - 1].get_node_list())
        size_l = len(layers[index].get_node_list())

    return np.random.randn(size_l, size_l_prev) * np.sqrt(1 / size_l_prev) * 30 - 3


def xavier(layers, index):
    if index == 0:
        return
    else:
        size_l_prev = len(layers[index - 1].get_node_list())
        size_l = len(layers[index].get_node_list())

    return np.random.randn(size_l, size_l_prev) * np.sqrt(2 / (size_l_prev + size_l)) * 30 - 3


def random(layers, index):
    if index == 0:
        return
    else:
        size_l_prev = len(layers[index - 1].get_node_list())
        size_l = len(layers[index].get_node_list())
        return (np.random.rand(size_l, size_l_prev) * 6) - 3
