import numpy as np
from matplotlib import pyplot as plt
import pandas as pd




def draw_plots(train_loss, train_acc, maxiter):




    step = max([1, int(maxiter / 20)])
    train_loss = train_loss[::step]
    train_acc = train_acc[::step]

    tl = pd.DataFrame(train_loss)
    ta = pd.DataFrame(train_acc)
    ta.to_csv('../../plots/ta.csv', index=False, index_label=False)
    tl.to_csv('../../plots/tl.csv', index=False, index_label=False)
    tl_len = len(train_loss)
    ta_len = len(train_acc)

    x_t = np.arange(tl_len)
    # x_v = np.arange(tl_len + 1, step=3)
    x2_t = np.arange(ta_len)
    # x2_v = np.arange(ta_len + 1, step=3)

    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 16,
            }

    fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2)
    fig.subplots_adjust(hspace=0.7)

    ax1.plot(x_t, train_loss, label='training')
    # ax1.plot(x_v, valid_loss, label='validation')
    ax1.legend()
    ax1.set_ylabel('Loss', fontdict=font)

    ax2.plot(x2_t, train_acc, label='training')
    # ax2.plot(x2_v, valid_acc, label='validation')
    ax2.legend()
    ax2.set_xlabel('Iterations', fontdict=font)
    ax2.set_ylabel('Accuracy', fontdict=font)

    fig.savefig(fname='../../plots/last_plot.jpg')

    plt.show()



if __name__ == '__main__':

    f = np.random.rand(50) * 6 - 3
    plt.scatter(np.arange(50), f)
