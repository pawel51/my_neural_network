from matplotlib.animation import FuncAnimation
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

matplotlib.use("TkAgg")

fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2)
fig.subplots_adjust(hspace=0.7)
xdata, yloss, yvloss, yaccu, yvaccu = [], [], [], [], []

ln_loss, = ax1.plot([], [])
ln_vloss, = ax1.plot([], [])
ln_accu, = ax2.plot([], [])
ln_vaccu, = ax2.plot([], [])

ITER = 90 #
DECREASE = 10 # times

def init():
    ax1.set_xlim(0, ITER)
    ax2.set_xlim(0, ITER)
    ax1.set_ylim(0, 0.3)
    ax2.set_ylim(0, 0.3)
    return ln_loss, ln_vloss, ln_accu, ln_vaccu,


def update(frame, tl, vl, ta, va):
    frame = int(frame)
    xdata.append(frame)
    yloss.append(tl[frame][0])
    yaccu.append(ta[frame][0])
    if frame != ITER-1:
        yvloss.append(vl[int(frame / 3)][0])
        yvaccu.append(va[int(frame / 3)][0])
    else:
        yvloss.append(vl[int(frame / 3)+1][0])
        yvaccu.append(va[int(frame / 3)+1][0])

    ln_loss.set_data(xdata, yloss)
    ln_vloss.set_data(xdata, yvloss)
    ln_accu.set_data(xdata, yaccu)
    ln_vaccu.set_data(xdata, yvaccu)
    return ln_loss, ln_vloss, ln_accu, ln_vaccu


def animate():
    tl = pd.read_csv('../../plots/tl.csv')
    vl = pd.read_csv('../../plots/vl.csv')
    ta = pd.read_csv('../../plots/ta.csv')
    va = pd.read_csv('../../plots/va.csv')
    tl = np.array(tl.iloc[:, 1:])
    vl = np.array(vl.iloc[:, 1:])
    ta = np.array(ta.iloc[:, 1:])
    va = np.array(va.iloc[:, 1:])

    ani = FuncAnimation(fig, update,
                        fargs=(tl, vl, ta, va),
                        frames=(np.linspace(0, ITER - 1, int((ITER-1)/DECREASE))).astype(dtype=int),
                        interval=100,
                        init_func=init,
                        blit=True,
                        cache_frame_data=True,
                        repeat=False)
    plt.show()
    ani.save(filename='anim')



if __name__ == '__main__':
    animate()
