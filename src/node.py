from edge import Edge


class Node:

    def __init__(self, r, k):
        self.r = r  # index of the node in the layer-k
        self.k = k  # index of the layer
        # initialize inputs
        self.inputs = []  # list of input edges
        # output value
        self.output = 0

    def create_inputs(self, prev_layer_node_list):
        for i in range(len(prev_layer_node_list)):
            self.inputs.append(Edge(prev_layer_node_list[i], self))

    def set_input_weights(self, weights):
        for input, index in zip(self.inputs, range(len(weights))):
            input.set_weight(weights[index])

    def set_output(self, number):
        self.output = number

    def get_output(self):
        return self.output

    def get_inputs(self):
        return self.inputs

    def to_string(self):
        node_str = ""
        for input in self.inputs:
            node_str += input.to_string()

        node_str += f"||r: {self.r} k:{self.k}||\n"
        node_str += f"OUTPUT: {self.output}\n"
        node_str += "----------\n"

        return node_str
