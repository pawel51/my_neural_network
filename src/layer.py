from node import Node
from myfunctions import he
import numpy as np


class Layer:
    def __init__(self, index, node_count):
        self.index = index
        self.node_count = node_count
        self.node_list = []
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

    def feed_layer(self, activ):
        for node in self.node_list:
            node.forward(activ)

    def start_back_prop(self, l_f, y, act):
        for node, i in zip(self.node_list, range(self.node_count)):
            node.start_back(l_f=l_f, y=y[i], act=act, outputs=self.get_outputs())

    def back_prop(self, act):
        for node in self.node_list:
            node.backward(act=act)
            node.set_error(0)

    def update_gradients(self, n, alfa, adam):
        if not adam:
            for node in self.node_list:
                node.update_gradients(n, alfa)
        if adam:
            for node in self.node_list:
                node.update_gradients_adam(n, alfa)

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
            a.append(round(node.get_output(), 4))
        return a

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
