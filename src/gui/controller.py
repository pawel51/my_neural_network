import tkinter as tk

import numpy as np
from tkinter.font import Font
from PIL import ImageTk, Image
from logic import config
from logic.main import run
import time

heart_attrs = [
    'age'
    , 'sex'
    , 'chest pain'  # chest pain type (4 values)
    , 'resting blood\npressure'  # resting blood pressure
    , 'serum cholestoral\nin mg/dl'  # serum cholestoral in mg/dl
    , 'fasting blood sugar'  # fasting blood sugar > 120 mg/dl
    , 'resting\ncardio'  # resting electrocardiographic results (values 0,1,2)
    , 'max heart rate'  # maximum heart rate achieved
    , 'exercise\ninduced angina'  # exercise induced angina
    , 'old peak'  # ST depression induced by exercise relative to rest
    , 'slope while\nexercise'  # the slope of the peak exercise ST segment
    , 'major vessels'  # number of major vessels (0-3) colored by flourosopy
    , 'blood disorder'  # thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
]

heart_labels = [
    'Had Heart\nAttack',
    'Did not Had\nHeart Attack'
]


class Controller:

    def __init__(self):
        pass

    def show_nrs(self, app):
        for i in range(len(app.nr_spin)):
            app.nr_spin[i].grid_remove()
            app.layer_lab[i].grid_remove()
        for i in range(int(app.laycnt_spinner.get())):
            app.nr_spin[i].grid()
            app.layer_lab[i].grid()
        print(f"Layers count{app.network_model.lay_count.get()}")

    def draw_model(self, app):
        print(list(app.network_canvas.find_all()))
        app.network_canvas.delete("net")
        x0 = app.network_canvas.winfo_width() / (int(app.network_model.lay_count.get()) + 1)
        y0s = []
        lay_sizes = []
        for i in range(int(app.network_model.lay_count.get())):
            # height of the canvas devided by number of neurons in the layer
            lay_sizes.append(int(app.network_model.neurons_spiners[i].get()))
            print(f"Lay size: {lay_sizes[i]}")
            y0s.append(app.network_canvas.winfo_height() / ((lay_sizes[i]) + 1))

        middles = []
        size = 20
        # draw circles
        # i number of layer
        for i in range(int(app.network_model.lay_count.get())):
            middles.append([])
            # j number of node
            for j in range(int(app.network_model.neurons_spiners[i].get())):
                x = x0 * (i + 1)
                y = y0s[i] * (j + 1)
                xhat = x0 * (i + 1) + size
                yhat = y0s[i] * (j + 1) + size

                # save the middle point
                middles[i].append((xhat - int(size / 2), yhat - int(size / 2)))

                app.network_canvas.create_oval(x, y, xhat, yhat,
                                               fill='#ffffff', width=3, tags=(i, j), tag=("net", f"l{i}n{j}"))
                # print(f"NODE: l:{i} l:{j}")

        # draw lines
        if len(middles) > 1:
            i = 1
            for lay0, lay1 in zip(middles, middles[1:]):
                k = 0
                for point0 in lay0:
                    j = 0
                    for point1 in lay1:
                        app.network_canvas.create_line(point0[0], point0[1], point1[0], point1[1],
                                                       smooth=True, splinestep=12, width=2,
                                                       tag=("net", f"l{i}n{j}e{k}"))
                        # print(f"EDGE: l{i}n{j}e{k}")
                        j += 1
                    k += 1
                print("\n")
                i += 1
            print("\n\n")
        print(app.network_canvas.find_all())

    def show_data(self, app, NET_WIDTH, NET_HEIGHT):
        # take all in attr and put them next to input layer

        x0 = 10
        print(app.network_canvas.winfo_width())
        print(app.network_canvas.winfo_height())
        xn = NET_WIDTH * 0.7
        y0 = NET_HEIGHT / (len(heart_attrs) + 1)
        for i in range(len(heart_attrs)):
            y = y0 * (i + 1)
            app.network_canvas.create_text((x0, y), anchor=tk.NW, text=heart_attrs[i], justify=tk.CENTER,
                                           font=Font(family='Helvetica', size=12, weight='bold'))

        y0 = NET_HEIGHT / (len(heart_labels) + 2)
        for i in range(len(heart_labels)):
            y = y0 * (i * 2 + 1)
            app.network_canvas.create_text((xn, y), anchor=tk.NW, text=heart_labels[i], justify=tk.CENTER,
                                           font=Font(family='Helvetica', size=13, weight='bold'))

    def update_train_plot(self, app):
        plot_png = Image.open('../../plots/last_plot.jpg')
        plot_png = plot_png.resize((480, 360))
        app.plot_tk = ImageTk.PhotoImage(plot_png)
        app.score_canvas.itemconfigure('img', image=app.plot_tk)

    def start_training(self, app, network):
        act_v = []
        neur_v = []
        for i in range(int(network.lay_count.get())):
            act_v.append(network.act.get())
            neur_v.append(int(network.neurons_spiners[i].get()))

        cb_id = app.after(2000, app.controller.update_colors, app)

        run(MB_SIZE=int(network.batch.get()),
            EPOCHS=int(network.epoch.get()),
            alfa=float(network.alfa.get()),
            activation_function=act_v,
            initializer=network.initializer.get(),
            loss_function=network.loss.get(),
            OPTIM=network.optim.get(),
            neuron_v=neur_v)


        app.after_cancel(cb_id)
        self.update_train_plot(app)

    def update_colors(self, app):
        with config.WEIGHTS_LOCK:
            lay_list = config.WEIGHTS
            if not lay_list:
                return

        i = len(lay_list)
        for layer in lay_list[::-1]:
            outs = 0
            for node in layer:
                outs += node[0]
            j = 0
            for node in layer:
                # <--- node output probability --->
                node[0] /= outs
                node[0] *= 510

                if node[0] > 255:
                    node[0] -= 255
                    node[0] = 255 - node[0]
                    a = np.base_repr(int(node[0]), base=16)
                    if node[0] < 16:
                        a = f'0{a}'
                    app.network_canvas.itemconfig(f'l{i}n{j}', fill=f'#FF{a}{a}')
                    # print(f'l{i}n{j} <-- #FF{a}{a}')
                else:
                    a = np.base_repr(int(node[0]), base=16)
                    if node[0] < 16:
                        a = f'0{a}'
                    app.network_canvas.itemconfig(f'l{i}n{j}', fill=f'#{a}FF{a}')
                    # print(f'l{i}n{j} <-- #{a}FF{a}')

                # <--- Edge Weights --->
                max_w = np.max(node[1:])
                min_w = np.abs(np.min(node[1:]))
                k = 0
                for weight in node[1:]:
                    weight += min_w
                    weight /= (max_w + min_w)
                    weight *= 510
                    weight = int(weight)

                    if weight > 255:
                        weight -= 255
                        weight = 255 - weight
                        a = np.base_repr(int(weight), base=16)
                        if weight < 16:
                            a = f'0{a}'
                        app.network_canvas.itemconfig(f'l{i}n{j}e{k}', fill=f'#FF{a}{a}')
                        # print(f'l{i}n{j}e{k} <-- #FF{a}{a}')
                    else:
                        a = np.base_repr(int(weight), base=16)
                        if weight < 16:
                            a = f'0{a}'
                        app.network_canvas.itemconfig(f'l{i}n{j}e{k}', fill=f'#{a}FF{a}')
                        # print(f'l{i}n{j}e{k} <-- #{a}FF{a}')
                    k += 1
                j += 1
            app.update_idletasks()
            i -= 1
        print("I AM RUNNING AND UPDATING COLORS MF")
        app.after(500, self.update_colors, app)
