from network import Network
import numpy as np
from time import time
from data_loader import load_data, split_arr
from plots import draw_plots
from saver import save_network
from numba import jit, cuda
from loader import load_network

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


# Mini batch size
MB_SIZE = 5
# Label size
LABEL_SIZE = 10
# Input Size
IN_SIZE = 784
# Number of Hidden layers, number of neurons in each
# HID_LAY = (1, 128)
# split rate
SPLIT_RATE = 0.75
# sample count
NUM = 60
# How many times through whole training set
EPOCHS = 10
# Optimizer 0-SGD, 1-ADAM, 2-SGD+velocity
OPTIM = 2


def run():
    start = time()
    # total steps count
    # maxiter = NUM*SPLIT_RATE/MB_SIZE*EPOCHS

    # <--- DATA --->
    # 20, 100, 1000, 10k, 60k
    train_data, valid_data = load_data(split_rate=SPLIT_RATE, num=NUM)

    print("DATA LOADED")

    # network = load_network('../networks/net1')
    # network = create_new_network(alfa, activation_function, initializer, loss_function, IN_SIZE, LABEL_SIZE)
    network = Network(
        alfa=0.001,
        initializer='xavier',
        loss_function='CE_v2')

    # First Layer
    network.append_layer(IN_SIZE, 'relu')
    network.append_layer(4, 'tanh')

    # Last Layer
    network.append_layer(LABEL_SIZE, 'tanh')

    network.concat_layers()
    network.init_weights()
    print('AFTER NETWORK INIT')

    # <--- Training Loop --->
    train_loss_sum = 0
    valid_loss_sum = 0
    train_accuracy = 0
    valid_accuracy = 0
    train_loss_arr = []
    valid_loss_arr = []
    train_accuracy_arr = []
    valid_accuracy_arr = []

    for epoch in range(EPOCHS):
        print(f"EPOCH {epoch} ")
        train_samples, train_labels = split_arr(array=train_data, mb_size=MB_SIZE)
        valid_samples, valid_labels = split_arr(array=valid_data, mb_size=MB_SIZE)
        # print("SPLIT DONE")
        iter = 0
        for train_batch_samples, train_batch_labels in zip(train_samples, train_labels):
            for train_sample, train_label in zip(train_batch_samples, train_batch_labels, ):
                train_loss_vector, outputs = network.train_sample(sample=train_sample[0],
                                                                  label=outs[train_label[0][0]]
                                                                  )
                print(outputs)
                train_loss_sum += np.sum(train_loss_vector)
                train_accuracy += np.exp(outputs[train_label[0][0]]) / np.sum(np.exp(outputs))

            network.update_gradients(MB_SIZE, opt=OPTIM)
            train_loss_arr.append(train_loss_sum / MB_SIZE)
            train_accuracy_arr.append(train_accuracy / MB_SIZE)
            print(f'train loss: {train_loss_sum / MB_SIZE}')
            print(f'train accu: {train_accuracy / MB_SIZE}')
            train_loss_sum = 0
            train_accuracy = 0
            iter += 1
            print(f"    Batch{iter}")


        for valid_batch_samples, valid_batch_labels in zip(valid_samples, valid_labels):
            for valid_sample, valid_label in zip(valid_batch_samples, valid_batch_labels):
                valid_loss_vector, outputs = network.test_sample(sample=valid_sample[0],
                                                                 label=outs[valid_label[0][0]],
                                                               )

                valid_loss_sum += np.sum(valid_loss_vector)

                valid_accuracy += np.exp(outputs[valid_label[0][0]]) / np.sum(np.exp(outputs))
                # print(f'LABEL: {valid_label[0][0]}, PREDICTION: {np.argmax(valid_loss_vector)}')
            valid_loss_arr.append(valid_loss_sum / MB_SIZE)
            valid_accuracy_arr.append(valid_accuracy / MB_SIZE)
            valid_accuracy = 0
            valid_loss_sum = 0

    print('AFTER TRAINING')
    # <--- Training ENDLoop --->
    end = time()

    print(f"Execution time: {round(end - start, 2)} s")

    save_network(network, '../networks/net_002')

    # print('AFTER TESTING')
    valid_loss_arr.insert(0, train_loss_arr[0])
    valid_accuracy_arr.insert(0, train_accuracy_arr[0])
    draw_plots(np.array(train_loss_arr), np.array(valid_loss_arr),
               np.array(train_accuracy_arr), np.array(valid_accuracy_arr))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()



