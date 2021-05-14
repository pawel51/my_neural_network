from edge import Edge
from layer import Layer


class Node:

    def __init__(self, r, k):
        self.r = r  # index in the layer
        self.k = k  # index of the layer
        # initialize inputs
        self.inputs = []  # list of input edges
        # output value
        self.output = 0

    def add_inputs_from_prev_layer(self, prev_layer):
        for prev_node in prev_layer.get_node_list():
            new_edge = Edge(prev_node, self)
            self.inputs.append(new_edge)

    def add_outputs_to_next_layer(self, next_layer):
        pass


    def get_output(self):
        return self.output

    def get_inputs(self):
        return self.inputs
