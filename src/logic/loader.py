from logic.network import Network
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
        initializer=atr['initializer'],
        loss_function=atr['loss_function'],
        opt=atr['optimalizer']
    )
    act_func = atr['activations']
    network.append_layer(atr['input_shape'], act_func[0])
    for i in range(atr['layers_count']):
        network.append_layer(atr['neuron_vector'][i], act_func[i])
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



def create_new_network(alfa, activations, initializer, loss_function, neuron_v, optim):
    network = Network(
        alfa=alfa,
        initializer=initializer,
        loss_function=loss_function,
        opt=optim)

    # First Layer
    j=0
    network.append_layer(neuron_v[0], '')
    j+=1
    for i in range(len(activations)-1):
        network.append_layer(neuron_v[i+1], activations[i])
        j+=1

    network.concat_layers()
    network.init_weights()
    print(f'AFTER NETWORK INIT, {j} Layers')
    return network


if __name__ == '__main__':
    load_network('../networks/net4')



