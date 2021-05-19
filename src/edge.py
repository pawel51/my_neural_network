

class Edge:
    def __init__(self, start, end):
        self.weight = 0  # before init weights
        self.start = start  # starting node address
        self.end = end  # ending node address
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

    def to_string(self):
        return f"{self.start.r}__{round(self.weight,2)}__{self.end.r} GRAD: {round(self.gradient, 2)}\n"
