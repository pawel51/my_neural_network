import tkinter as tk
from tkinter import ttk
from tkinter.font import Font
from gui.controller import Controller
from gui.network_model import Network_model
from PIL import ImageTk, Image
import threading


WIDTH = 1400
HEIGHT = 800
NET_WIDTH = WIDTH * 0.56
NET_HEIGHT = HEIGHT * 0.74
PARAMS_BG = '#ffbbff'
NET_BG = '#666666'
MAIN_BG = '#bbEEFF'
BTN_BG = '#62d2e1'
SCORE_BG = '#54bede'
MAX_LAY_CNT = 4
MAX_NEURONS = 13


class Application(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.master.title("NN - Creator")
        self.grid(sticky=tk.N + tk.S + tk.E + tk.W)
        self.master.title('Sample application')
        self.master.geometry = self.center(HEIGHT, WIDTH)
        self.main_canvas = 0
        self.param_canvas = 0
        self.layers_canvas = 0
        self.network_canvas = 0
        self.score_canvas = 0
        self.alfa_spinner = 0
        self.act_combox = 0
        self.loss_combox = 0
        self.init_combox = 0
        self.opotim_combox = 0
        self.epoch_spinner = 0
        self.batch_spinner = 0
        self.laycnt_spinner = 0
        self.train_btn = 0
        self.draw_btn = 0
        self.load_entry = 0
        self.load_checkbox = 0
        self.save_entry = 0
        self.save_checkbox = 0
        self.lay_btn = 0
        self.plot_tk = None
        self.top_level1 = 0
        self.progress_bar = 0
        self.nr_spin = []
        self.layer_lab = []
        self.network_model = Network_model()
        self.controller = Controller()
        self.createWidgets()

    def createWidgets(self):
        top = self.winfo_toplevel()

        top.rowconfigure(0, weight=1)

        top.columnconfigure(0, weight=1)

        self.rowconfigure(0, weight=1)

        self.columnconfigure(0, weight=1)

        self.main_canvas = tk.Canvas(self, height=HEIGHT, width=WIDTH, bg=MAIN_BG)
        self.main_canvas.grid(row=0, column=0, sticky=tk.N + tk.S + tk.E + tk.W)
        self.main_canvas.grid_propagate(0)

        # <--- Parameter canvas --->
        self.param_canvas = tk.Canvas(self.main_canvas, height=HEIGHT, width=WIDTH * 0.2, bg=PARAMS_BG)
        self.param_canvas.grid(row=0, column=0, sticky=tk.W)
        self.param_canvas.grid_propagate(0)
        # <--- Container 1 canvas --->
        container1 = tk.Canvas(self.main_canvas, height=HEIGHT, width=0.6 * WIDTH, bg=PARAMS_BG)
        container1.grid(row=0, column=1, sticky=tk.N + tk.S + tk.E + tk.W)
        # <--- Layers canvas --->
        self.layers_canvas = tk.Canvas(container1, height=HEIGHT * 0.16, width=WIDTH * 0.56, bg=PARAMS_BG)
        self.layers_canvas.grid(row=0, column=1)
        self.layers_canvas.grid_propagate(0)
        # <--- Network canvas --->
        self.network_canvas = tk.Canvas(container1, height=NET_HEIGHT, width=NET_WIDTH, bg=NET_BG)
        self.network_canvas.grid(row=1, column=1)
        self.network_canvas.grid_propagate(0)

        # <--- Alfa Spinner --->
        alfa_label = tk.Label(self.param_canvas, anchor='e', padx=10, pady=10, text="Alfa", bg=PARAMS_BG,
                              font=Font(family='Helvetica', size=20, weight='bold'))
        alfa_label.grid(row=0, column=0, padx=10, pady=10)
        self.alfa_spinner = tk.Spinbox(self.param_canvas, justify=tk.CENTER, from_=0.001, increment=0.001, to=0.999,
                                       width=5, repeatinterval=25, font=Font(family='Helvetica', size=20),
                                       textvariable=self.network_model.alfa)
        self.alfa_spinner.grid(row=0, column=1, padx=10, pady=10)
        # <--- Activation ComBox --->
        act_label = tk.Label(self.param_canvas, anchor='e', padx=10, pady=10, text="Activation\n function",
                             bg=PARAMS_BG,
                             font=Font(family='Helvetica', size=10, weight='bold'))
        act_label.grid(row=1, column=0, padx=10, pady=10)
        self.act_combox = ttk.Combobox(self.param_canvas, values=['sigmoid', 'tanh', 'relu', 'leaky_relu'], width=10,
                                       textvariable=self.network_model.act)
        self.act_combox.current(0)
        self.act_combox.grid(row=1, column=1, padx=10, pady=10)
        # <--- Loss Function ComBox --->
        loss_label = tk.Label(self.param_canvas, anchor='e', padx=10, pady=10, text="Loss\n function",
                              bg=PARAMS_BG,
                              font=Font(family='Helvetica', size=10, weight='bold'))
        loss_label.grid(row=2, column=0, padx=10, pady=10)
        self.loss_combox = ttk.Combobox(self.param_canvas, values=['L1', 'L2', 'Cross Entropy'], width=10,
                                        textvariable=self.network_model.loss)
        self.loss_combox.current(0)
        self.loss_combox.grid(row=2, column=1, padx=10, pady=10)
        # <--- Init weights ComBox --->
        init_label = tk.Label(self.param_canvas, anchor='e', padx=10, pady=10, text="Initializing\n function",
                              bg=PARAMS_BG,
                              font=Font(family='Helvetica', size=10, weight='bold'))
        init_label.grid(row=3, column=0, padx=10, pady=10)
        self.init_combox = ttk.Combobox(self.param_canvas, values=['random', 'Xavier', 'He'], width=10,
                                        textvariable=self.network_model.initializer)
        self.init_combox.current(0)
        self.init_combox.grid(row=3, column=1, padx=10, pady=10)
        # <--- Optimize ComBox --->
        optim_label = tk.Label(self.param_canvas, anchor='e', padx=10, pady=10, text="Optimizing\n function",
                               bg=PARAMS_BG,
                               font=Font(family='Helvetica', size=10, weight='bold'))
        optim_label.grid(row=4, column=0, padx=10, pady=10)
        self.optim_combox = ttk.Combobox(self.param_canvas, values=['sgd', 'sgd_momentum', 'ADAM'], width=10,
                                         textvariable=self.network_model.optim)
        self.optim_combox.current(0)
        self.optim_combox.grid(row=4, column=1, padx=10, pady=10)
        # <--- Epoch Spinner --->
        epoch_label = tk.Label(self.param_canvas, anchor='e', padx=10, pady=10, text="Epochs\n Count",
                               bg=PARAMS_BG,
                               font=Font(family='Helvetica', size=10, weight='bold'))
        epoch_label.grid(row=5, column=0, padx=10, pady=10)
        self.epoch_spinner = tk.Spinbox(self.param_canvas, justify=tk.CENTER, from_=1, increment=1, to=10000,
                                        width=5, repeatinterval=15, font=Font(family='Helvetica', size=20),
                                        textvariable=self.network_model.epoch)
        self.epoch_spinner.grid(row=5, column=1, padx=10, pady=10)
        # <--- BatchSize Spinner --->
        batch_label = tk.Label(self.param_canvas, anchor='e', padx=10, pady=10, text="Batch\n size",
                               bg=PARAMS_BG,
                               font=Font(family='Helvetica', size=10, weight='bold'))
        batch_label.grid(row=6, column=0, padx=10, pady=10)
        self.batch_spinner = tk.Spinbox(self.param_canvas, justify=tk.CENTER, from_=1, increment=1, to=100,
                                        width=5, repeatinterval=25, font=Font(family='Helvetica', size=20),
                                        textvariable=self.network_model.batch)
        self.batch_spinner.grid(row=6, column=1, padx=10, pady=10)
        # <--- Load network with name of --->
        load_label = tk.Label(self.param_canvas, anchor='e', padx=10, pady=10, text="Load from: ",
                              bg=PARAMS_BG,
                              font=Font(family='Helvetica', size=10, weight='bold'))
        load_label.grid(row=7, column=0, padx=10, pady=10)
        self.load_entry = tk.Entry(self.param_canvas, font=Font(family='Helvetica', size=12), width=15,
                                   textvariable=self.network_model.load_path)
        self.load_entry.grid(row=7, column=1, pady=10)
        self.load_checkbox = tk.Checkbutton(self.param_canvas, anchor=tk.E, bg=PARAMS_BG,
                                            variable=self.network_model.load_bool)
        self.load_checkbox.grid(row=7, column=2)
        # <--- Save Network as name --->
        save_label = tk.Label(self.param_canvas, anchor='e', padx=10, pady=10, text="Save as: ",
                              bg=PARAMS_BG,
                              font=Font(family='Helvetica', size=10, weight='bold'))
        save_label.grid(row=8, column=0, padx=10, pady=10)
        self.save_entry = tk.Entry(self.param_canvas, font=Font(family='Helvetica', size=12), width=15,
                                   textvariable=self.network_model.save_path)
        self.save_entry.grid(row=8, column=1, pady=10)
        self.save_checkbox = tk.Checkbutton(self.param_canvas, anchor=tk.E, activebackground='#001345', bg=PARAMS_BG,
                                            variable=self.network_model.save_bool)
        self.save_checkbox.grid(row=8, column=2)
        # <--- Init Network and draw model on canvas --->
        self.draw_btn = tk.Button(self.param_canvas, anchor='e', padx=5, pady=5, text="Init model",
                                  font=Font(family='Helvetica', size=12, weight='bold'), bg=BTN_BG,
                                  command=lambda: [self.network_model.init(self),
                                                   self.controller.draw_model(self)])
        self.draw_btn.grid(row=9, column=0, padx=10, pady=10)
        # <--- Start Training --->
        self.train_btn = tk.Button(self.param_canvas, anchor='e', padx=5, pady=5, text="Train model",
                                   font=Font(family='Helvetica', size=12, weight='bold'), bg=BTN_BG,
                                   command=lambda: [threading.Thread(target=self.controller.start_training,
                                                                     args=(app, self.network_model)).start()])
        self.train_btn.grid(row=9, column=1, columnspan=1, padx=10, pady=10)

        # <--- Number of layers spinner --->
        laycnt_label = tk.Label(self.layers_canvas, anchor='e', padx=10, pady=10, text="Layers\n Count", bg=PARAMS_BG,
                                font=Font(family='Helvetica', size=12, weight='bold'))
        laycnt_label.grid(row=0, column=0, padx=10, pady=10)
        self.laycnt_spinner = tk.Spinbox(self.layers_canvas, justify=tk.LEFT, from_=1, increment=1, to=MAX_LAY_CNT,
                                         textvariable=self.network_model.lay_count, width=2, repeatinterval=200,
                                         font=Font(family='Helvetica', size=16))
        self.laycnt_spinner.grid(row=0, column=1, padx=10, pady=10)

        '''
        # <--- Button to apply layers change --->
        '''
        self.lay_btn = tk.Button(self.layers_canvas, bd=4, bg=BTN_BG, text='Apply',
                                 font=Font(family='Helvetica', size=12),
                                 command=lambda: [self.network_model.init(self),
                                                  self.controller.show_nrs(self)])
        self.lay_btn.grid(row=1, column=0, columnspan=2, padx=10)
        for i in range(MAX_LAY_CNT):
            neuron_label = tk.Label(self.layers_canvas, anchor=tk.CENTER, padx=10, pady=10, text="ID",
                                    bg=PARAMS_BG, font=Font(family='Helvetica', size=12, weight='bold'))
            neuron_label.grid(row=0, column=2 + i, padx=10, pady=10)
            neuron_label.grid_remove()
            neuron_spinner = tk.Spinbox(self.layers_canvas, justify=tk.LEFT, from_=1, increment=1, to=MAX_NEURONS,
                                        width=2, repeatinterval=20, font=Font(family='Helvetica', size=16),
                                        textvariable=tk.IntVar())
            neuron_spinner.grid(row=1, column=2 + i, padx=10, pady=10)
            neuron_spinner.grid_remove()
            self.nr_spin.append(neuron_spinner)
            self.layer_lab.append(neuron_label)
        self.nr_spin[0].grid()
        self.layer_lab[0].grid()

        # <--- Atribute names and Labels --->
        self.controller.show_data(self, NET_WIDTH, NET_HEIGHT)

        # <--- Progress Bar --->
        pb = Image.open('../../images/progressbar.png')
        self.progress_bar = ImageTk.PhotoImage(pb)
        self.network_canvas.create_image((0.54 * WIDTH, 0.07 * HEIGHT), anchor=tk.NW, image=self.progress_bar, tag='bar')
        (x1, y1, x2, y2) = self.network_canvas.bbox('bar')
        self.network_canvas.create_text((x1+8,y1 + 10), text='1', font=Font(family='Helvetica', size=16))
        self.network_canvas.create_text((x1+8,y2 - 16), text='0', font=Font(family='Helvetica', size=16))


        # <--- Training plot window --->

        self.top_level1 = tk.Toplevel(width=480, height=360, padx=10, pady=10)
        self.top_level1.title("Training plots")
        self.score_canvas = tk.Canvas(self.top_level1, height=360, width=480)
        self.score_canvas.grid(row=0, column=0)
        plot_png = Image.open('../../images/first_plot.png')
        self.plot_tk = ImageTk.PhotoImage(plot_png)
        self.score_canvas.create_image((0, 0), anchor=tk.NW, image=self.plot_tk, tag="img")


    def center(self, game_height, game_width):
        window_height = self.master.winfo_height()
        window_width = self.master.winfo_width()
        screen_width = int(self.master.winfo_screenwidth())
        screen_height = int(self.master.winfo_screenheight())
        print(f"{screen_width}\n{screen_height}")
        x = int((screen_width / 2) - (game_width / 2))
        y = int((screen_height / 2) - (game_height / 2))
        return f"{window_width}x{window_height}+{x}+{y}"


if __name__ == '__main__':
    app = Application()
    app.mainloop()
