from network import Network
import numpy as np
from time import time
from data_loader import load_data

categories = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
    ]

outs = [
    [1,0,0,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,0,1],
]

def print_index_of_max(vector):
    return

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start = time()
    layers = []

    # <--- DATA --->
    samples, labels = load_data()

    network = Network(
        alfa=0.4,
        activation_function='sigmoid',
        initializer='he',
        loss_function='L2')

    # First Layer
    network.append_layer(784)

    sample0 = samples[0]
    labels0 = labels[0]
    print(labels0)
    for i in range(1, 10):
        network.append_layer(10)

    # Last Layer
    network.append_layer(10)

    network.concat_layers()
    network.init_weights()
    print('AFTER INIT')


    for i in range(100):
        network.train_sample(sample=samples[i%10], label=outs[labels[i%10][0]])
        if i % 2 == 0:
            network.update_gradients(2, adam=0)

    print('AFTER TRAINING')


    y = []
    y.append(network.test_sample(sample=sample0))


    print('AFTER TESTING')

    y = np.array(y)
    max_index_of_y = np.argmax(y)
    print(max_index_of_y)
    print(f'Should be: {labels0[0]}')
    end = time()

    print(f"Execution time: {round(end-start, 2)} s")

