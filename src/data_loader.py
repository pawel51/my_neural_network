import numpy as np
import pandas as pd


def load_data():
    data_train = pd.read_csv('../images/train/fashion-mnist_train.csv')
    data_test = pd.read_csv('fashion-mnist_test.csv')
    # 60.000 of images
    train_images = np.array(data_train.iloc[:, 1:]) / 255
    # 60.000 of acording labels
    train_labels = np.array(data_train.iloc[:, 0])
    # 784 pixels
    # img0 = float(images[0])
    # label0 = labels[0]
    X_test = np.array(data_test.iloc[:, 1:])
    test_labels = np.array(data_train.iloc[:, 0])