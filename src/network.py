from myfunctions import relu, he
from layer import Layer
from edge import Edge

class Network:
    def __init__(self, alfa, activation_function, layers, initializer, data):
        self.alfa = alfa
        self.layers = layers
        self.layers_len = len(layers)
        self.initializer = initializer
        if activation_function == 'relu':
            self.act_func = relu
        self.inputs = data



    def concat_layers(self):
        i_layer=0
        for layer in self.layers:
            for node in layer.get_node_list():
                if i_layer == 0:
                    node.add_inputs_from_user(self.layers[0], self.inputs)
                if i_layer > 0:
                    node.add_inputs_from_prev_layer(self.layers[i_layer - 1])
            i_layer += 1

    def init_weights(self, initializer):
        i_layer = 0
        j_node = 0
        k_input = 0
        for layer in self.layers:
            weights = he(layers=self.layers, index=i_layer)
            for node in layer.get_node_list():
                for input in node.get_inputs():
                    input.set_weight(weights[j_node][k_input])
                    k_input += 1
                j_node += 1
        i_layer += 1


