import numpy as np
import pandas as pd


img_rows, img_cols = 28, 28


def get_data(num):
    data_test = pd.read_csv('../images/train/fashion-mnist_train.csv')

    train_images = np.array(data_test.iloc[:,:])
    # train_labels = np.array(data_test.iloc[:,0:1])

    train_images = np.array(train_images)[:num]
    # train_labels = np.array(train_labels)[:num]
    df = pd.DataFrame(train_images)
    # df2 = pd.DataFrame(train_labels)
    df.to_csv(f'../images/train/{num}_train.csv')
    # df2.to_csv(f'../images/train/{num}_labels.csv')


def load_data(split_rate=0.75, num=20):
    # train_valid_labels = pd.read_csv(f'../images/train/{num}_labels.csv')
    train_valid_samples = pd.read_csv(f'../images/train/{num}_train.csv')
    # 60.000 of images
    train_valid_samples = np.array(train_valid_samples.iloc[:,1:])
    # df = pd.DataFrame(train_images)
    # df.to_csv('../images/train/10_train.csv')
    # 60.000 of acording labels
    # train_valid_labels = np.array(train_valid_labels.iloc[:,1:])

    ts = int(train_valid_samples.shape[0] * split_rate)

    train_samples = train_valid_samples[:ts, :]
    valid_samples = train_valid_samples[ts:, :]
    # train_labels = train_valid_labels[:ts, :]
    # valid_labels = train_valid_labels[ts:, :]

    return train_samples, valid_samples



    # df = pd.DataFrame(train_labels)
    # df.to_csv('../images/train/10_labels.csv')

    # 784 pixels
    # img0 = float(images[0])
    # label0 = labels[0]
    # X_test = np.array(data_test.iloc[:, 1:])
    # test_labels = np.array(data_train.iloc[:, 0])


def split_arr(mb_size, array):
    b_arr = []

    rng = np.random.default_rng()
    for i in range(int(len(array) / mb_size)):
        mb_arr = []
        for j in range(mb_size):
            index = rng.choice(len(array), size=1)
            mb_arr.append(array[index])
            array = np.delete(array, index, axis=0)
        b_arr.append(np.array(mb_arr.copy()))
        del mb_arr
    if array:
        np.append(array)

    b_arr = np.array(b_arr) / 255
    return b_arr[:,:,:,1:], b_arr[:,:,:,0:1]


if __name__ == '__main__':
    # get_data(20)
    train_data, valid_data = load_data(split_rate=0.75, num=20)
    train_samples, train_labels = split_arr(array=train_data, mb_size=5)
    valid_data = split_arr(array=valid_data, mb_size=5)
    print('ok')