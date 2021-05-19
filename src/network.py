from myfunctions import relu, he, sigmoid, tanh, xavier, L2


class Network:
    def __init__(self, alfa, activation_function, layers, initializer, loss_function, data):
        self.alfa = alfa
        self.layers = layers
        self.layers_len = len(layers)

        self.inputs = data
        if initializer == 'he':
            self.initializer = he
        if initializer == 'xavier':
            self.initializer = xavier
        else:
            self.initializer = he

        self.initializer = initializer
        if activation_function == 'relu':
            self.act_func = relu
        if activation_function == 'sigmoid':
            self.act_func = sigmoid
        if activation_function == 'tanh':
            self.act_func = tanh
        else:
            self.act_func = relu

        if loss_function == 'L2':
            self.l_f = L2

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

    def train_sample(self, estimator):
        self.feed_sample()
        self.start_back_prop(estimator=estimator)
        self.back_prop()

    def feed_sample(self):
        self.layers[0].add_inputs_from_user(self.inputs)
        for layer in self.layers[1:]:
            layer.feed_layer(self.act_func)

    def start_back_prop(self, estimator):
        self.layers[len(self.layers)-1].start_back_prop(l_f=self.l_f, y=estimator, act=self.act_func)


    def back_prop(self):
        for layer in self.layers[self.layers_len:0:-1]:
            layer.back_prop(act=self.act_func)

    def print_network(self):
        for layer in self.layers:
            print(layer.to_string())
