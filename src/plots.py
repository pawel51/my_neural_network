import numpy as np
from matplotlib import pyplot as plt
import pandas as pd




def draw_plots(train_loss, valid_loss, train_acc, valid_acc):
    tl = pd.DataFrame(train_loss)
    vl = pd.DataFrame(valid_loss)
    ta = pd.DataFrame(train_acc)
    va = pd.DataFrame(valid_acc)
    tl.to_csv('../plots/tl.csv')
    vl.to_csv('../plots/vl.csv')
    ta.to_csv('../plots/ta.csv')
    va.to_csv('../plots/va.csv')

    tl_len = len(train_loss)
    ta_len = len(train_acc)

    x_t = np.arange(tl_len)
    x_v = np.arange(tl_len + 1, step=3)
    x2_t = np.arange(ta_len)
    x2_v = np.arange(ta_len + 1, step=3)

    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 16,
            }

    fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2)
    fig.subplots_adjust(hspace=0.7)

    ax1.plot(x_t, train_loss, label='training')
    ax1.plot(x_v, valid_loss, label='validation')
    ax1.legend()
    ax1.set_title('Loss function', fontdict=font)
    ax1.set_xlabel('Iterations', fontdict=font)
    ax1.set_ylabel('Loss', fontdict=font)

    ax2.plot(x2_t, train_acc, label='training')
    ax2.plot(x2_v, valid_acc, label='validation')
    ax2.legend()
    ax2.set_title('Accuracy function ', fontdict=font)
    ax2.set_xlabel('Iterations', fontdict=font)
    ax2.set_ylabel('Accuracy', fontdict=font)

    plt.show()


