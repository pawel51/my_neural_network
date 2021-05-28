from myfunctions import relu, he, sigmoid, tanh, xavier, L2, L1, BCE
from layer import Layer
import numpy as np

class Network:
    def __init__(self, alfa, activation_function, initializer, loss_function):
        self.alfa = alfa
        self.layers = []
        self.layers_len = 0


        if initializer == 'he':
            self.initializer = he
        if initializer == 'xavier':
            self.initializer = xavier


        self.initializer = initializer
        if activation_function == 'relu':
            self.act_func = relu
        if activation_function == 'sigmoid':
            self.act_func = sigmoid
        if activation_function == 'tanh':
            self.act_func = tanh


        if loss_function == 'L1':
            self.l_f = L1

        if loss_function == 'L2':
            self.l_f = L2

        if loss_function == 'BCE':
            self.l_f = BCE

    def append_layer(self, neurons_num):
        self.layers.append(Layer(self.layers_len, node_count=neurons_num))
        self.layers_len += 1

    def concat_layers(self):
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

    def update_gradients(self, n, adam=0):
        # n how many samples in training iteration
        # first layer is input layer so dont update it
        for layer in self.layers[1:]:
            layer.update_gradients(n, self.alfa, adam=adam)

    def calc_loss(self):
        # yhat = np.array(self.layers[len(self.layers) - 1].get_outputs())
        # y = np.array(label)
        # accumulated loss vector
        loss_v = np.array(self.layers[len(self.layers)-2].get_losses())
        return loss_v

    # <--- Training starts here --->
    def train_sample(self, sample, label):
        self.feed_sample(sample)
        self.start_back_prop(estimator=label)
        self.back_prop()
        return self.calc_loss()

    # <--- Testing starts here --->
    def test_sample(self, sample, label):
        self.feed_sample(sample)
        return self.calc_loss()

    def feed_sample(self, inputs):
        self.layers[0].add_inputs_from_user(inputs)
        for layer in self.layers[1:]:
            layer.feed_layer(self.act_func)

    def start_back_prop(self, estimator):
        self.layers[len(self.layers)-1].start_back_prop(l_f=self.l_f, y=estimator, act=self.act_func)


    def back_prop(self):
        for layer in self.layers[len(self.layers):0:-1]:
            layer.back_prop(act=self.act_func)

    def print_network(self):
        for layer in self.layers:
            print(layer.to_string())
