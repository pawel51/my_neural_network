from network import Network
import numpy as np
from time import time
from data_loader import load_data, split_arr
from plots import loss_plot, accuracy_plot

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
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
]

# Numer of iterations
NITER = 3
# Mini batch size
MB_SIZE = 5
# Label size
LABEL_SIZE = 10
# Input Size
IN_SIZE = 784
# Number of Hidden layers, number of neurons in each
HID_LAY = (3, 10)
# split rate
SPLIT_RATE = 0.75
# sample count
NUM = 20

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start = time()

    # <--- DATA --->
    # 10, 20, 100, 1000, 10k, 60k
    train_data, valid_data = load_data(split_rate=SPLIT_RATE, num=NUM)
    train_data = split_arr(array=train_data, mb_size=MB_SIZE)
    valid_data = split_arr(array=valid_data, mb_size=MB_SIZE)
    tra

    network = Network(
        alfa=0.1,
        activation_function='sigmoid',
        initializer='he',
        loss_function='L2')

    # First Layer
    network.append_layer(IN_SIZE)

    for i in range(1, HID_LAY[0] - 1):
        network.append_layer(HID_LAY[1])

    # Last Layer
    network.append_layer(LABEL_SIZE)

    network.concat_layers()
    network.init_weights()
    print('AFTER INIT')

    train_loss_sum = 0
    valid_loss_sum = 0
    train_loss_arr = []
    valid_loss_arr = []
    train_accuracy_arr = []
    valid_accuracy_arr = []
    for i in range(1, NITER * NUM + 1):

        train_sample = train_samples[i % int(NUM * SPLIT_RATE)]
        train_label = outs[train_labels[i % int(NUM * SPLIT_RATE)][0]]
        valid_sample = valid_samples[i % int(NUM * (1 - SPLIT_RATE))]
        valid_label = outs[valid_labels[i % int(NUM * (1 - SPLIT_RATE))][0]]
        train_loss_vector = network.train_sample(sample=train_sample,
                                                 label=train_label
                                                 )
        valid_loss_vector = network.test_sample(sample=valid_sample,
                                                label=valid_label
                                                )
        train_loss_sum += np.sum(train_loss_vector)
        valid_loss_sum += np.sum(valid_loss_vector)

        # TODO
        # acc = corect / sum(all) *100%
        corr_train_index = train_labels[i % int(NUM * SPLIT_RATE)][0]
        corr_valid_index = valid_labels[i % int(NUM * (1 - SPLIT_RATE))][0]
        train_accuracy = train_loss_vector[corr_train_index] / np.sum(train_loss_vector) * 100
        valid_accuracy = valid_loss_vector[corr_valid_index] / np.sum(train_loss_vector) * 100

        network.update_gradients(MB_SIZE, adam=0)
        train_loss_arr.append(train_loss_sum)
        valid_loss_arr.append(valid_loss_sum)

        train_accuracy_arr.append(train_accuracy)
        valid_accuracy_arr.append(valid_accuracy)

        train_loss_sum = 0
        valid_loss_sum = 0

    print('AFTER TRAINING')

    # print('AFTER TESTING')
    loss_plot(np.array(train_loss_arr), np.array(valid_loss_arr), np.array(train_accuracy_arr), np.array(valid_accuracy_arr))

    end = time()

    print(f"Execution time: {round(end - start, 2)} s")
