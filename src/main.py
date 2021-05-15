from layer import Layer
from network import Network
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from time import time

img_rows, img_cols = 28, 28
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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start = time()
    layers = []
    layers.append(Layer(0, 2))


    for i in range(1, 1000):
        layers.append(Layer(i, 100))


    network = Network(0.3, 'relu', layers, 'he', ['0','1'])
    network.concat_layers()
    end = time()

    print(f"{round(end-start, 2)} s")
    # print(network.print_network())
