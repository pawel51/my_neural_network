from edge import Edge


class Node:

    def __init__(self, r, k):
        self.r = r  # index of the node in the layer-k
        self.k = k  # index of the layer
        # initialize inputs
        self.inputs = []  # list of input edges
        # output value
        self.output = 0
        self.bias = 0
        self.bias_gradient = 0
        self.error = 0

    def create_inputs(self, prev_layer_node_list):
        for i in range(len(prev_layer_node_list)):
            self.inputs.append(Edge(prev_layer_node_list[i], self))

    def set_input_weights(self, weights):
        for input, index in zip(self.inputs, range(len(weights))):
            input.set_weight(weights[index])

    def forward(self, activ):
        sigma = 0
        for input in self.inputs:
            x = input.get_start().get_output()
            w = input.get_weight()
            sigma += float(x) * w
        sigma += self.bias
        self.output = activ(sigma)

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
            input.set_gradient(dyhat_dw)
            self.bias_gradient = dyhat_dsigma
            prev_node.add_error(self.error*input.get_weight())

    def start_back(self, l_f, y, act):
        # dL/dyhat
        dL_dyhat = l_f(y=y, yhat=self.output, derivative=1)
        self.error = dL_dyhat
        self.loop_back_inputs(act=act)

    def backward(self, act):
        self.loop_back_inputs(act)


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

    def to_string(self):
        node_str = ""
        for input in self.inputs:
            node_str += input.to_string()
        node_str += f"Bias: {self.bias} b_GRAD: {round(self.bias_gradient, 2)}"
        node_str += f"||r: {self.r} k:{self.k}||\n"
        node_str += f"OUTPUT: {round(self.output, 2)}\n"
        node_str += "----------\n"

        return node_str
