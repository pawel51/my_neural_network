from layer import Layer
from network import Network
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt


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
    layers = []
    layers.append(Layer(0, 2))
    layers.append(Layer(1, 3))
    layers.append(Layer(2, 1))



    network = Network(0.3, 'relu', layers)
