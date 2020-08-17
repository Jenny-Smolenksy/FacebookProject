#import torch
#import torch.nn as nn
import numpy as np
import tensorflow
from torch.utils.data import TensorDataset, DataLoader

#import pandas

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# change from names od Type to numbers
def normType(dataset):
    # run on all data lines and change the second column - normalize between 1 to 4
    dataset[:, 1][dataset[:, 1] == 'Link'] = 1
    dataset[:, 1][dataset[:, 1] == 'Photo'] = 2
    dataset[:, 1][dataset[:, 1] == 'Status'] = 3
    dataset[:, 1][dataset[:, 1] == 'Video'] = 4


# change from Type from number to names
def normNumToType(dataset):
    # run on all data lines and change the second column - normalize between 1 to 4
    dataset[:, 1][dataset[:, 1] == '1'] = 'Link'
    dataset[:, 1][dataset[:, 1] == '2'] = 'Photo'
    dataset[:, 1][dataset[:, 1] == '3'] = 'Status'
    dataset[:, 1][dataset[:, 1] == '4'] = 'Video'

#get data set and change every range to number between 1-6
def normLikesToRange(dataset):
    dataset[:, 16][dataset[:, 16] < 50] = 1
    dataset[:, 16][(dataset[:, 16] >= 50) & (dataset[:, 16] < 100)] = 2
    dataset[:, 16][(dataset[:, 16] >= 100) & (dataset[:, 16] < 200)] = 3
    dataset[:, 16][(dataset[:, 16] >= 200) & (dataset[:, 16] < 500)] = 4
    dataset[:, 16][dataset[:, 16] >= 500] = 5


def main():
    #for data - 16 is like
    filename = 'data\dataset_Facebook.csv'
    data = np.loadtxt(filename, dtype=str, delimiter=";")

    #delete first row - name
    data = np.delete(data, (0), axis=0)

    #delete lines with missing value - like
    data =  np.delete(data, (111), axis=0)
    data = np.delete(data, (119), axis=0)
    data = np.delete(data, (122), axis=0)
    data = np.delete(data, (161), axis=0)
    data = np.delete(data, (495), axis=0)

    #norm the type
    normType(data)

    data = data.astype(np.int)

    #norm likes range
    normLikesToRange(data)

    # shuffle randim lines
    np.random.shuffle(data)

    num_of_rows = data[:, 0].size
    # here 80 percent of train
    train_size = 0.8 * num_of_rows
    train_size = int(train_size)

    data_train = data[0:train_size]
    data_test = data[train_size:num_of_rows]

    #take the likes as targets
    train_y = data_train[:,16]
    test_y = data_test[:,16]

    #delte it from features
    train_x = np.delete(data_train, 16 , axis=1)
    test_x = np.delete(data_test, 16 , axis=1)

    tensor_x = torch.Tensor(train_x)  # transform to torch tensor
    tensor_y = torch.Tensor(train_y)

    my_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
    my_dataloader = DataLoader(my_dataset)  # create your dataloader

    #norm back to the type string
#    data_train = data_train.astype(np.str)
#    data_test = data_test.astype(np.str)

#    normNumToType(data_train)
#    normNumToType(data_test)

    #save train and test file
    #np.savetxt('train.csv', data_train, fmt='%s', delimiter=';')
    #np.savetxt('test.csv', data_test, fmt='%s', delimiter=';')

    print("HELLO")


if __name__ == '__main__':
    main()