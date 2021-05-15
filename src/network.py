from myfunctions import relu, he


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
        # self.layers[0].add_inputs_from_user(self.inputs)
        prev_layer = self.layers[0]
        for layer in self.layers[1:]:
            layer.concat_layers(prev_layer)
            prev_layer = layer

    def init_weights(self):
        i_layer = 1
        for layer in self.layers[1:]:
            weights = he(layers=self.layers, index=i_layer)
            layer.init_weights(weights)
            i_layer += 1

    def feed_sample(self):
        self.layers[0].add_inputs_from_user(self.inputs)
        for layer in self.layers[1:]:
            layer.feed_layer(self.act_func)







    def print_network(self):
        for layer in self.layers:
            print(layer.to_string())


