import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate


def draw_plots(train_loss, valid_loss, train_acc, valid_acc):
    tl_len = len(train_loss)
    ta_len = len(train_acc)


    x_t = np.arange(tl_len)
    x_v = np.arange(tl_len+1, step=3)
    x2_t = np.arange(ta_len)
    x2_v = np.arange(ta_len+1, step=3)



    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 16,
            }

    plt.plot(x_t, train_loss, x_v, valid_loss)
    plt.title('Loss function', fontdict=font)
    plt.xlabel('Iterations', fontdict=font)
    plt.ylabel('Loss', fontdict=font)
    plt.show()

    plt.plot(x2_t, train_acc, x2_v, valid_acc)
    plt.title('Accuracy function', fontdict=font)
    plt.xlabel('Iterations', fontdict=font)
    plt.ylabel('Accuracy (%)', fontdict=font)
    plt.show()


