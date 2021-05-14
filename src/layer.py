from node import Node
from myfunctions import he


class Layer:
    def __init__(self, index, node_count):
        self.index = index
        self.node_list = [node_count]
        # create all nodes in the layer init: (index of node, index of the layer)
        for i in range(node_count):
            self.node_list.append(Node(i, index))




    def get_node_list(self):
        return self.node_list
