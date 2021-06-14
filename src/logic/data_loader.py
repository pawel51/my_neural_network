import numpy as np
import pandas as pd



def get_data(num, path1, path2):
    data_test = pd.read_csv(path1)

    train_images = np.array(data_test.iloc[:,:])
    # train_labels = np.array(data_test.iloc[:,0:1])

    train_images = np.array(train_images)[:num]
    # train_labels = np.array(train_labels)[:num]
    df = pd.DataFrame(train_images)
    df.fillna(0)
    # df2 = pd.DataFrame(train_labels)
    df.to_csv(path2, index=False, index_label=False)
    # df2.to_csv(f'../images/train/{num}_labels.csv')


def normalize(arr):

    arr = arr.T

    for atr in arr:
        max = np.max(atr)
        atr /= max
    return arr.T

def split_data(split, arr):
    id1 = int(arr.shape[0] * split*0.5)
    id2 = int(arr.shape[0] - arr.shape[0] * split*0.5)
    test_data0 = arr[:id1,:]
    test_data1 = arr[id2:,:]
    test_data = np.concatenate((test_data0, test_data1))
    train_data = arr[id1:id2,:]
    return train_data, test_data





def load_data_hearts(path='../../heart_data/train/303_heart.csv', split=0.2):
    train_test_samples = pd.read_csv(path)
    arr = normalize(np.array(train_test_samples.iloc[:,:], dtype=np.float64))

    train_data, test_data = split_data(split, arr)

    return train_data, test_data




def randomize(mb_size, array):

    mb_arr = np.empty(shape=(0, array.shape[1]))
    rng = np.random.default_rng()
    for i in range(int(len(array) / mb_size)):
        for j in range(mb_size):
            index = rng.choice(len(array), size=1)
            mb_arr = np.append(mb_arr, array[int(index[0]):int(index[0])+1,:], axis=0)
            array = np.delete(array, index, axis=0)

    return mb_arr


if __name__ == '__main__':
    get_data(303, '../heart_data/train/heart.csv', '../heart_data/train/303_heart.csv')
    # load_data_hearts()





    # train_data, valid_data = load_data(split_rate=0.75, num=1000)
    # train_samples, train_labels = split_arr(array=train_data, mb_size=10)
    # valid_data = split_arr(array=valid_data, mb_size=5)
    print('ok')