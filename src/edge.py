import math as m

class Edge:
    def __init__(self, start, end):
        self.weight = 0  # before init weights
        self.start = start  # starting node address
        self.end = end  # ending node address
        self.gradient = 0
        self.mean = 0.1
        self.variance = 0.1
        self.v = 10


    def update_gradient(self, n, alfa):
        self.weight -= alfa * (self.gradient/n)
        self.gradient = 0

    def update_gradient_momentum(self, n, alfa):
        beta = 0.9

        self.gradient /= n

        self.v = self.v*beta - alfa * self.gradient
        self.weight += self.v
        self.gradient = 0


    def update_gradient_adam(self, n, alfa):
        beta1 = 0.9
        beta2 = 0.999
        eps = 0.00000001

        self.gradient /= n

        self.mean = beta1 * self.mean - (1 - beta1) * self.gradient
        mean = self.mean / (1 - beta1)

        self.variance = beta2 * self.variance + (1 - beta2) * self.gradient * self.gradient
        variance = self.variance / (1 - beta2)


        self.weight -= alfa * (mean / (m.sqrt(variance)+eps))
        self.gradient = 0


    def set_gradient(self, gradient):
        self.gradient = gradient

    def add_to_gradient(self, num):
        self.gradient += num

    def get_gradient(self):
        return self.gradient

    def get_end(self):
        return self.end

    def get_start(self):
        return self.start

    def get_weight(self):
        return self.weight

    def set_end(self, new_end):
        self.end = new_end

    def set_start(self, new_start):
        self.start = new_start

    def set_weight(self, new_weight):
        self.weight = new_weight

    def add_to_weight(self, num):
        self.weight += num

    def to_string(self):
        return f"{self.start.r}__{round(self.weight,2)}__{self.end.r} GRAD: {round(self.gradient, 2)}\n"
