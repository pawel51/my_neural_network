import numpy as np
import pandas as pd


img_rows, img_cols = 28, 28



def load_data():
    train_labels = pd.read_csv('../images/train/10_labels.csv')
    train_samples = pd.read_csv('../images/train/10_train.csv')
    # data_test = pd.read_csv('fashion-mnist_test.csv')
    # 60.000 of images
    train_samples = np.array(train_samples.iloc[:, 1:])
    # df = pd.DataFrame(train_images)
    # df.to_csv('../images/train/10_train.csv')
    # 60.000 of acording labels
    train_labels = np.array(train_labels.iloc[:,1:])


    return train_samples, train_labels



    # df = pd.DataFrame(train_labels)
    # df.to_csv('../images/train/10_labels.csv')

    print("ala")
    # 784 pixels
    # img0 = float(images[0])
    # label0 = labels[0]
    # X_test = np.array(data_test.iloc[:, 1:])
    # test_labels = np.array(data_train.iloc[:, 0])


if __name__ == '__main__':
    load_data()
