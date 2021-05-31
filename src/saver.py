import numpy as np
from network import Network
import pandas as pd


def load_network():



def save_network(network, path):
    net, weights, nodes, layers, atr = [], [], [], [], []
    atr.append(
        (network.alfa,
         network.act_func,
         network.initializer,
         network.l_f)
    )

    for layer in network.layers:
        for node in layer.node_list:
            for edge in node.inputs:
                weights.append(edge.get_weight())
            nodes.append(weights)
        layers.append(nodes)

    net.append(atr)
    net.append(layers)

    network_arr = np.array(layers)

    df = pd.DataFrame(network_arr)

    df.to_csv(f'networks/{path}')


if __name__ == '__main__':
    network = Network(
        alfa=0.1,
        activation_function='relu',
        initializer='he',
        loss_function='CEL')

    # First Layer
    network.append_layer(10)

    network.append_layer(128)

    # Last Layer
    network.append_layer(10)

    network.concat_layers()
    network.init_weights()

    save_network(network, '../networks/net1.csv')
