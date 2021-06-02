from network import Network
import json
import csv

def load_network(path):
    lines = []
    # Global strings
    atr = {}

    with open(f'{path}_atr.json', newline='') as infile:
        atr.update(json.load(infile))

    print(atr)

    network = Network(
        alfa=atr['alfa'],
        activation_function=atr['activation'],
        initializer=atr['initializer'],
        loss_function=atr['loss_function']
    )

    network.append_layer(atr['input_shape'])
    for i in range(atr['layers_count']):
        network.append_layer(atr['neuron_vector'][i])
    network.concat_layers()
    # print(network.print_network())
    i_row = 0
    with open(f'{path}.csv', newline='') as infile:
        reader = csv.reader(infile, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            lines.append(row)

        n_vars = lines.pop()
        n_means = lines.pop()
        n_vels = lines.pop()
        n_biases = lines.pop()

    for layer in network.layers[1:]:
        for node in layer.node_list:
            node.bias = n_biases.pop(0)
            node.v = n_vels.pop(0)
            node.mean = n_means.pop(0)
            node.variance = n_vars.pop(0)
            weights = lines[i_row]
            e_vels = lines[i_row + 1]
            e_means = lines[i_row + 2]
            e_vars = lines[i_row + 3]
            i_row += 4
            for edge in node.inputs:
                edge.weight = weights.pop(0)
                edge.v = e_vels.pop(0)
                edge.mean = e_means.pop(0)
                edge.variance = e_vars.pop(0)

    # print(network.print_network())

    return network



def create_new_network(alfa, activation_function, initializer, loss_function, insize, outsize):
    network = Network(
        alfa=alfa,
        activation_function=activation_function,
        initializer=initializer,
        loss_function=loss_function)

    # First Layer
    network.append_layer(insize)
    network.append_layer(128)
    # Last Layer
    network.append_layer(outsize)

    network.concat_layers()
    network.init_weights()
    print('AFTER NETWORK INIT')
    return network


if __name__ == '__main__':
    load_network('../networks/net1')



