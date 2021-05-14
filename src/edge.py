from myfunctions import he


class Edge:
    def __init__(self, start, end):
        self.weight = 0  # random
        self.start = start  # starting node address
        self.end = end  # ending node address

    def get_end(self):
        return self.end

    def get_start(self):
        return self.start

    def get_weigth(self):
        return self.weight

    def set_start(self, new_start):
        self.start = new_start

    def set_end(self, new_end):
        self.end = new_end

    def set_weight(self, new_weight):
        self.weight = new_weight


