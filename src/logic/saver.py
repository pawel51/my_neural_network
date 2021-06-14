import json
import csv


def save_network(network, path):
    try:
        with open(f'{path}.csv', "r+") as loc:
            loc.truncate(0)
        loc.close()
    except FileNotFoundError:
        pass

    weights, e_vels, e_means, e_vars = [], [], [], []
    n_biases, n_vels, n_means, n_vars = [], [], [], []
    # Global strings
    atr = {'alfa': network.alfa,
           'activations': [],
           'initializer': network.initializer,
           'loss_function': network.loss_function,
           'input_shape': len(network.layers[0].node_list),
           'layers_count': len(network.layers[1:]),
           'neuron_vector': [],
           'optimalizer': network.opt
           }
    # EDGE
    # self.mean = 0
    # self.variance = 0
    # self.v = 0
    # NODE
    # self.output = 0
    # self.bias = 0
    # self.bias_grad = 0
    # self.error = 0
    # self.mean = 0
    # self.variance = 0
    # self.loss = 0
    # self.v = 0

    for layer in network.layers[1:]:
        atr['neuron_vector'].append(len(layer.node_list))
        atr['activations'].append(layer.activation)
        for node in layer.node_list:
            for edge in node.inputs:
                weights.append(edge.weight)
                e_vels.append(edge.v)
                e_means.append(edge.mean)
                e_vars.append(edge.variance)
            write_to_file(e_means, e_vars, e_vels, path, weights)
            clear_arrays(e_means, e_vars, e_vels, weights)
            n_biases.append(node.bias)
            n_vels.append(node.v)
            n_means.append(node.mean)
            n_vars.append(node.variance)
    write_to_file(n_means, n_vars, n_vels, path, n_biases)
    clear_arrays(n_means, n_vars, n_vels, n_biases)

    #     layers.append(nodes)
    # net.append(layers)
    # network_arr = np.array(layers)
    with open(f'{path}_atr.json', 'w') as outfile:
        json.dump(atr, outfile)


def clear_arrays(e_means, e_vars, e_vels, weights):
    weights.clear()
    e_vels.clear()
    e_means.clear()
    e_vars.clear()


def write_to_file(e_means, e_vars, e_vels, path, weights):
    with open(f'{path}.csv', 'a', newline='') as csvfile:
        buffer = csv.writer(csvfile, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
        buffer.writerow(weights)
        buffer.writerow(e_vels)
        buffer.writerow(e_means)
        buffer.writerow(e_vars)
    csvfile.close()


if __name__ == '__main__':


    save_network(network, '../networks/net1')
