from node import Node
from myfunctions import he


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
            print("ERROR: INPUT SIZE DIFFERS FROM LAYER[0] LEANGTH")

        for node, index in zip(self.node_list, range(len(data_list))):
            node.set_output(data_list[index])

    def concat_layers(self, prev_layer):
        for node in self.node_list:
            node.create_inputs(prev_layer.get_node_list())

    def init_weights(self, weights):
        for node, index in zip(self.node_list, range(self.node_count)):
            node.set_input_weights(weights[index])

    def to_string(self):
        layer_str = ""
        if self.index == 0:
            layer_str = "******************\n"
        layer_str += f"L: {self.index}\n"

        if self.index == 0:
            for node in self.node_list:
                layer_str += "OUTPUT: "+str(node.get_output()) + "\n"
            layer_str += "******************\n"
            return layer_str

        for node in self.node_list:
            layer_str += node.to_string()
        layer_str += "******************\n"
        return layer_str

    def get_node_list(self):
        return self.node_list
