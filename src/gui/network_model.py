import tkinter as tk



class Network_model:

    def __init__(self):
        self.load_bool = tk.IntVar()
        self.save_bool = tk.IntVar()
        self.load_path = tk.StringVar()
        self.save_path = tk.StringVar()
        self.act = tk.StringVar()
        self.loss = tk.StringVar()
        self.initializer = tk.StringVar()
        self.lay_count = tk.IntVar()
        self.neurons_spiners = []
        self.alfa = tk.StringVar()
        self.optim = tk.StringVar()
        self.batch = tk.StringVar()
        self.epoch = tk.StringVar()

    def init(self, app):
        self.neurons_spiners.clear()
        for i in range(int(self.lay_count.get())):
            self.neurons_spiners.append(app.nr_spin[i])










