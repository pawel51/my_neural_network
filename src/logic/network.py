from logic.myfunctions import he, xavier, L2, L1, BCE, CEL, CE_v2, random
from logic.layer import Layer
import numpy as np


class Network:
    def __init__(self, alfa, initializer, loss_function, opt):
        self.optimalizer = opt
        self.initializer = initializer
        # defaults
        self.alfa = alfa
        self.init = random
        self.l_f = L2
        self.layers = []
        self.layers_len = 0
        self.opt = opt
        self.loss_function = loss_function

        self.set_opt(opt)
        self.set_l_f(loss_function)
        self.set_initializer(initializer)

    def append_layer(self, neurons_num, act_func):
        self.layers.append(Layer(self.layers_len, node_count=neurons_num, activation_function=act_func))
        self.layers_len += 1

    def concat_layers(self):
        prev_layer = self.layers[0]
        for layer in self.layers[1:]:
            layer.concat_layers(prev_layer)
            prev_layer = layer

    def init_weights(self):
        i_layer = 1
        for layer in self.layers[1:]:
            weights = self.init(layers=self.layers, index=i_layer)
            layer.init_weights(weights)
            i_layer += 1

    def update_gradients(self, n):
        # n how many samples in training iteration
        # first layer is input layer so dont update it
        for layer in self.layers[1:]:
            layer.update_gradients(n, self.alfa, opt=self.opt)

    # <--- Training starts here --->
    def train_sample(self, sample, label):
        self.feed_sample(sample)
        self.layers[len(self.layers) - 1].start_back_prop(l_f=self.l_f, y=label)
        for layer in self.layers[len(self.layers)-2:1:-1]:
            layer.back_prop()
        if label[0] == 1:
            i=0
        else:
            i=1
        outputs = self.layers[len(self.layers) - 1].get_outputs()
        loss = self.layers[len(self.layers) - 1].get_node_list()[i].get_loss()
        accu = np.exp(outputs[i]) / np.sum(np.exp(outputs))
        return loss, accu

    # <--- Testing starts here --->
    def test_sample(self, sample, label):
        self.feed_sample(sample)
        return self.layers[len(self.layers) - 1].get_outputs()

    def feed_sample(self, inputs):
        self.layers[0].add_inputs_from_user(inputs)
        for layer in self.layers[1:]:
            layer.feed_layer()

    def back_prop(self):
        for layer in self.layers[len(self.layers):0:-1]:
            layer.back_prop()

    def print_network(self):
        for layer in self.layers:
            print(layer.to_string())

    def get_weights(self):
        weights = []
        for layer in self.layers[1:]:
            weights.append(layer.get_weights())
        return weights
        # init

    def set_l_f(self, l_f):
        if l_f == 'L1':
            self.l_f = L1

        if l_f == 'L2':
            self.l_f = L2

        if l_f == 'BCE':
            self.l_f = BCE

        if l_f == 'CEL':
            self.l_f = CEL

        if l_f == 'Cross Entropy':
            self.l_f = CE_v2

    def set_alfa(self, al):
        self.alfa = al

    def set_opt(self, opt):
        if opt == 'sgd':
            self.opt = 0
        if opt == 'ADAM':
            self.opt = 1
        if opt == 'sgd_momentum':
            self.opt = 2

    def set_initializer(self, initializer):
        if initializer == 'he':
            self.init = he
        if initializer == 'xavier':
            self.init = xavier
        if initializer == 'random':
            self.init = random
