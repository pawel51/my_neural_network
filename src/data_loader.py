import numpy as np
import pandas as pd


img_rows, img_cols = 28, 28


def get_data(num):
    data_test = pd.read_csv('../images/train/fashion-mnist_train.csv')

    train_images = np.array(data_test.iloc[:,1:])
    train_labels = np.array(data_test.iloc[:,0:1])

    train_images = np.array(train_images)[:num] /255
    train_labels = np.array(train_labels)[:num]
    df = pd.DataFrame(train_images)
    df2 = pd.DataFrame(train_labels)
    df.to_csv(f'../images/train/{num}_train.csv')
    df2.to_csv(f'../images/train/{num}_labels.csv')

    print("")

def load_data(split_rate=0.75, num=10):
    train_valid_labels = pd.read_csv(f'../images/train/{num}_labels.csv')
    train_valid_samples = pd.read_csv(f'../images/train/{num}_train.csv')
    # 60.000 of images
    train_valid_samples = np.array(train_valid_samples.iloc[:, 1:])
    # df = pd.DataFrame(train_images)
    # df.to_csv('../images/train/10_train.csv')
    # 60.000 of acording labels
    train_valid_labels = np.array(train_valid_labels.iloc[:,1:])

    ts = int(train_valid_samples.shape[0] * split_rate)

    train_samples = train_valid_samples[:ts, :]
    valid_samples = train_valid_samples[ts:, :]
    train_labels = train_valid_labels[:ts, :]
    valid_labels = train_valid_labels[ts:, :]

    return train_samples, train_labels, valid_samples, valid_labels



    # df = pd.DataFrame(train_labels)
    # df.to_csv('../images/train/10_labels.csv')

    # 784 pixels
    # img0 = float(images[0])
    # label0 = labels[0]
    # X_test = np.array(data_test.iloc[:, 1:])
    # test_labels = np.array(data_train.iloc[:, 0])


if __name__ == '__main__':
    get_data(10)
