from edge import Edge
import math as m
import numpy as np

class Node:

    def __init__(self, r, k):
        self.r = r  # index of the node in the layer-k
        self.k = k  # index of the layer
        # initialize inputs
        self.inputs = []  # list of input edges
        # output value
        self.output = 0
        self.bias = 0
        self.bias_grad = 0
        self.error = 0
        self.mean = 0
        self.variance = 0
        self.loss = 0

    def create_inputs(self, prev_layer_node_list):
        for node in prev_layer_node_list:
            self.inputs.append(Edge(node, self))


    def set_input_weights(self, weights):
        for input, index in zip(self.inputs, range(len(weights))):
            input.set_weight(weights[index])

    def forward(self, activ):
        sigma = 0
        for input in self.inputs:
            x = input.get_start().get_output()
            w = input.get_weight()
            sigma += float(x) * w

        self.set_output(activ(sigma+self.bias))


    def loop_back_inputs(self, act):
        for input in self.inputs:
            # prev node
            prev_node = input.get_start()
            # dyhat/dsigma
            dyhat_dsigma = self.error * act(self.output, derivative=1)
            # dsigma/dw
            dsigma_dw = prev_node.get_output()
            # dyhat/dw = dyhat/dsigma * dsigma/dw
            dyhat_dw = dyhat_dsigma * dsigma_dw
            input.add_to_gradient(dyhat_dw)
            self.bias_grad += dyhat_dsigma
            prev_node.add_error(self.error * input.get_weight())

    def start_back(self, l_f, y, act, outputs):
        # dL/dyhat
        dL_dyhat = l_f(y=y, yhat=self.get_output(), outputs=np.array(outputs), derivative=1)
        self.loss = l_f(y=y, yhat=self.get_output(), outputs=np.array(outputs), derivative=0)
        self.error = dL_dyhat
        self.loop_back_inputs(act=act)

    def backward(self, act):
        self.loop_back_inputs(act)

    def update_gradients(self, n, alfa):
        self.bias -= alfa * (self.bias_grad / n)
        self.bias_grad = 0
        for input_ in self.inputs:
            input_.update_gradient(n=n, alfa=alfa)

    def update_gradients_adam(self, n, alfa):
        beta1 = 0.9
        beta2 = 0.999
        eps = 0.00000001

        # self.bias_grad /= n

        self.mean = beta1 * self.mean - (1 - beta1) * self.bias_grad
        mean = self.mean / (1 - beta1)

        self.variance = beta2 * self.variance + (1 - beta2) * self.bias_grad * self.bias_grad
        variance = self.variance / (1 - beta2)

        self.bias -= alfa * (mean / (m.sqrt(variance) + eps))
        self.bias_grad = 0
        for input_ in self.inputs:

            input_.update_gradient_adam(n=n, alfa=alfa)


    def get_loss(self):
        return self.loss

    def get_error(self):

        return self.error


    def set_error(self, error):
        self.error = error


    def add_error(self, num):
        self.error += num


    def set_output(self, number):
        self.output = number


    def get_output(self):
        return self.output


    def add_to_output(self, num):
        self.output += num


    def get_inputs(self):
        return self.inputs

    def get_bias(self):
        return self.bias

    def get_bias_grad(self):
        return self.bias_grad

    def to_string(self):
        node_str = ""
        for input in self.inputs:
            node_str += input.to_string()
        node_str += f"Bias: {round(self.get_bias(), 2)} b_GRAD: {round(self.get_bias_grad(), 2)}"
        node_str += f"||r: {self.r} k:{self.k}||\n"
        node_str += f"OUTPUT: {round(self.get_output(), 4)}\n"
        node_str += "----------\n"

        return node_str
