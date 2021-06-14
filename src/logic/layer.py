from logic.node import Node
import numpy as np
from logic.myfunctions import relu, sigmoid, tanh, leaky_relu

class Layer:
    def __init__(self, index, node_count, activation_function):
        self.index = index
        self.node_count = node_count
        self.node_list = []
        self.activation = activation_function

        if activation_function == 'relu':
            self.act_func = relu
        if activation_function == 'sigmoid':
            self.act_func = sigmoid
        if activation_function == 'tanh':
            self.act_func = tanh
        if activation_function == 'leaky_relu':
            self.act_func = leaky_relu

        # create all nodes in the layer init: (index of node, index of the layer)
        for i in range(node_count):
            self.node_list.append(Node(i, index))

    def add_inputs_from_user(self, data_list):
        if self.node_count != len(data_list) and self.index == 0:
            print(self.node_count)
            print("ERROR: INPUT SIZE DIFFERS FROM LAYER[0] LIST")

        for node, index in zip(self.node_list, range(len(data_list))):
            node.set_output(data_list[index])

    def concat_layers(self, prev_layer):
        for node in self.node_list:
            node.create_inputs(prev_layer.get_node_list())

    def init_weights(self, weights):
        for node, index in zip(self.node_list, range(self.node_count)):
            node.set_input_weights(weights[index])

    def feed_layer(self):
        for node in self.node_list:
            node.forward(self.act_func)

    def start_back_prop(self, l_f, y):
        for node, i in zip(self.node_list, range(self.node_count)):
            node.start_back(l_f=l_f, y=y[i], act=self.act_func, outputs=self.get_outputs())

    def back_prop(self):
        for node in self.node_list:
            node.loop_back_inputs(act=self.act_func)
            node.set_error(0)


    def update_gradients(self, n, alfa, opt):
        if opt == 0:
            for node in self.node_list:
                node.update_gradients(n, alfa)
        if opt == 1:
            for node in self.node_list:
                node.update_gradients_adam(n, alfa)
        if opt == 2:
            for node in self.node_list:
                node.update_gradients_momentum(n, alfa)

    def get_weights(self):
        weights = []
        for node in self.node_list:
            weights.append(node.get_weights())
        return weights


    # returns losses vector
    def get_losses(self):
        a = []
        for node in self.node_list:
            a.append((node.get_loss()))
        return a

    # returns output vector
    def get_outputs(self):
        a = []
        for node in self.node_list:
            a.append(node.get_output())
        return np.array(a)

    def to_string(self):
        layer_str = ""
        if self.index == 0:
            layer_str = "******************\n"
        layer_str += f"L: {self.index}\n"

        if self.index == 0:
            for node in self.node_list:
                layer_str += "OUTPUT: " + str(node.get_output()) + "\n"
            layer_str += "******************\n"
            return layer_str

        for node in self.node_list:
            layer_str += node.to_string()
        layer_str += "******************\n"
        return layer_str

    def get_node_list(self):
        return self.node_list
