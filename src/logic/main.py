import numpy as np
import time
from logic.data_loader import load_data_hearts, randomize
from logic.plots import draw_plots
from logic.saver import save_network
from logic.loader import load_network, create_new_network
import threading
from logic import config

heart_attrs = [
    'age'
    , 'sex'
    , 'cp'  # chest pain type (4 values)
    , 'trestbps'  # resting blood pressure
    , 'chol'  # serum cholestoral in mg/dl
    , 'fbs'  # fasting blood sugar > 120 mg/dl
    , 'restecg'  # resting electrocardiographic results (values 0,1,2)
    , 'thalach'  # maximum heart rate achieved
    , 'exang'  # exercise induced angina
    , 'oldpeak'  # ST depression induced by exercise relative to rest
    , 'slope'  # the slope of the peak exercise ST segment
    , 'ca'  # number of major vessels (0-3) colored by flourosopy
    , 'thal'  # thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
]

heart_outs = [
    [1, 0],  # Had heart attack
    [0, 1]  # Didnt
]

heart_labels = [
    'Had Heart\nAttack',
    'Did not Had\nHeart Attack'
]


# Label size
LABEL_SIZE = 2
# Input Size
IN_SIZE = 13
# Number of Hidden layers, number of neurons in each
# HID_LAY = (1, 128)
# split rate
SPLIT_RATE = 0.2
# sample count
NUM = 303


VISION = 1


# Optimizer 0-SGD, 1-ADAM, 2-SGD+velocity


def run(MB_SIZE, EPOCHS, alfa, activation_function, initializer, loss_function, OPTIM, neuron_v):
    # neuron_v.insert(0, IN_SIZE)
    # neuron_v.append(LABEL_SIZE)

    start = time.time()
    # total steps count
    maxiter = int(NUM * (1 - SPLIT_RATE) / MB_SIZE * EPOCHS)

    train_data, test_data = load_data_hearts(split=0.2)

    print("DATA LOADED")

    # network = load_network('../networks/net_002')

    # alfa = 0.5
    # activation_function = ['tanh', 'tanh', 'relu']
    # initializer = 'xavier'
    # loss_function = 'CV_v2'
    # OPTIM = 'adam'
    network = create_new_network(alfa=-1 * alfa,
                                 activations=activation_function,
                                 initializer=initializer,
                                 loss_function=loss_function,
                                 neuron_v=neuron_v,
                                 optim=OPTIM)
    if VISION == 1:
        with config.WEIGHTS_LOCK:
            config.WEIGHTS = network.get_weights()
    time.sleep(0.5)

    # network.print_network()

    train_loss_sum = 0
    train_accuracy = 0
    train_loss_arr = []
    train_accuracy_arr = []
    i = 0
    batch_cnt = int(NUM * (1 - SPLIT_RATE) / MB_SIZE)
    for epoch in range(EPOCHS):
        train_samples = randomize(mb_size=MB_SIZE, array=train_data)
        for batch in range(batch_cnt):
            for sam in range(MB_SIZE):
                sample = train_samples[i:i + 1, :train_samples.shape[1] - 1][0]
                label = train_samples[i:i + 1, train_samples.shape[1] - 1:][0]
                loss, accu = network.train_sample(sample=sample,
                                                  label=heart_outs[int(label)])
                i += 1

                train_loss_sum += loss
                # if outputs[int(label)] == np.max(outputs):
                #     train_accuracy += 1
                train_accuracy += accu

            print(f"Batch nr {batch + 1} done")
            network.update_gradients(MB_SIZE)
            if VISION == 1:
                with config.WEIGHTS_LOCK:
                    config.WEIGHTS = network.get_weights()
            time.sleep(1)
            train_loss_arr.append(train_loss_sum / MB_SIZE)
            train_accuracy_arr.append(train_accuracy / MB_SIZE)
            print(f'train loss: {train_loss_sum / MB_SIZE}')
            print(f'train accu: {train_accuracy / MB_SIZE}')
            train_loss_sum = 0
            train_accuracy = 0
        i = 0
        print(f'Epoch {epoch + 1} done')
    print('AFTER TRAINING')
    draw_plots(np.array(train_loss_arr), np.array(train_accuracy_arr), maxiter)

    end = time.time()

    print(f"Execution time: {round(end - start, 2)} s")

    save_network(network, '../../networks/net_002')

    # print('AFTER TESTING')
    # valid_loss_arr.insert(0, train_loss_arr[0])
    # valid_accuracy_arr.insert(0, train_accuracy_arr[0])


def debug_run():
    # neuron_v.insert(0, IN_SIZE)
    # neuron_v.append(LABEL_SIZE)

    start = time.time()
    # total steps count


    train_data, test_data = load_data_hearts(split=0.2)

    print("DATA LOADED")

    # network = load_network('../networks/net_002')

    EPOCHS = 1
    MB_SIZE = 10
    alfa = 0.5
    activation_function = ['tanh', 'tanh', 'relu']
    initializer = 'xavier'
    loss_function = 'CV_v2'
    OPTIM = 'sgd'
    neuron_v = [IN_SIZE, 5, LABEL_SIZE]
    network = create_new_network(alfa=alfa,
                                 activations=activation_function,
                                 initializer=initializer,
                                 loss_function=loss_function,
                                 neuron_v=neuron_v,
                                 optim=OPTIM)
    if VISION == 1:
        with config.WEIGHTS_LOCK:
            config.WEIGHTS = network.get_weights()
    time.sleep(0.5)

    # network.print_network()

    train_loss_sum = 0
    train_accuracy = 0
    train_loss_arr = []
    train_accuracy_arr = []
    i = 0
    batch_cnt = int(NUM * (1 - SPLIT_RATE) / MB_SIZE)
    for epoch in range(EPOCHS):
        train_samples = randomize(mb_size=MB_SIZE, array=train_data)
        for batch in range(batch_cnt):
            for sam in range(MB_SIZE):
                sample = train_samples[i:i + 1, :train_samples.shape[1] - 1][0]
                label = train_samples[i:i + 1, train_samples.shape[1] - 1:][0]
                loss, accu = network.train_sample(sample=sample,
                                                  label=heart_outs[int(label)])
                i += 1

                train_loss_sum += loss
                # if outputs[int(label)] == np.max(outputs):
                #     train_accuracy += 1
                train_accuracy += accu

            print(f"Batch nr {batch + 1} done")
            network.update_gradients(MB_SIZE)
            if VISION == 1:
                with config.WEIGHTS_LOCK:
                    config.WEIGHTS = network.get_weights()
            time.sleep(1)
            train_loss_arr.append(train_loss_sum / MB_SIZE)
            train_accuracy_arr.append(train_accuracy / MB_SIZE)
            print(f'train loss: {train_loss_sum / MB_SIZE}')
            print(f'train accu: {train_accuracy / MB_SIZE}')
            train_loss_sum = 0
            train_accuracy = 0
        i = 0
        print(f'Epoch {epoch + 1} done')
    print('AFTER TRAINING')

    maxiter = int(NUM * (1 - SPLIT_RATE) / MB_SIZE * EPOCHS)
    draw_plots(np.array(train_loss_arr), np.array(train_accuracy_arr), maxiter)

    end = time.time()

    print(f"Execution time: {round(end - start, 2)} s")


    save_network(network, '../../networks/net_002')
    if VISION == 1:
        with config.WEIGHTS_LOCK:
            config.WEIGHTS = None

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    debug_run()
