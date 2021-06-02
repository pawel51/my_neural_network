from myfunctions import he, xavier, L2, L1, BCE, CEL, CE_v2, random
from layer import Layer
import numpy as np


class Network:
    def __init__(self, alfa, initializer, loss_function):
        self.alfa = alfa
        self.layers = []
        self.layers_len = 0

        self.initializer = initializer
        if initializer == 'he':
            self.init = he
        if initializer == 'xavier':
            self.init = xavier
        if initializer == 'random':
            self.init = random

        self.loss_function = loss_function

        if loss_function == 'L1':
            self.l_f = L1

        if loss_function == 'L2':
            self.l_f = L2

        if loss_function == 'BCE':
            self.l_f = BCE

        if loss_function == 'CEL':
            self.l_f = CEL

        if loss_function == 'CE_v2':
            self.l_f = CE_v2

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

    def update_gradients(self, n, opt=0):
        # n how many samples in training iteration
        # first layer is input layer so dont update it
        for layer in self.layers[1:]:
            layer.update_gradients(n, self.alfa, opt=opt)

    def calc_loss(self):
        # yhat = np.array(self.layers[len(self.layers) - 1].get_outputs())
        # y = np.array(label)
        # accumulated loss vector
        loss_v = np.array(self.layers[len(self.layers) - 1].get_losses())
        return loss_v

    # <--- Training starts here --->
    def train_sample(self, sample, label):
        self.feed_sample(sample)
        self.layers[len(self.layers) - 1].start_back_prop(l_f=self.l_f, y=label)
        self.back_prop()
        return self.calc_loss(), self.layers[len(self.layers) - 1].get_outputs()

    # <--- Testing starts here --->
    def test_sample(self, sample, label):
        self.feed_sample(sample)
        return self.calc_loss(), self.layers[len(self.layers) - 1].get_outputs()

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
