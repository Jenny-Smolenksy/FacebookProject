import numpy as np


def delete_empty_rows(data):
    # delete first row - name
    data = np.delete(data, (0), axis=0)

    # delete lines with missing value - like
    data = np.delete(data, (111), axis=0)
    data = np.delete(data, (119), axis=0)
    data = np.delete(data, (122), axis=0)
    data = np.delete(data, (161), axis=0)
    data = np.delete(data, (495), axis=0)



# change from names od Type to numbers
def type_to_number(dataset):
    # run on all data lines and change the second column - normalize between 1 to 4
    dataset[:, 1][dataset[:, 1] == 'Link'] = 1
    dataset[:, 1][dataset[:, 1] == 'Photo'] = 2
    dataset[:, 1][dataset[:, 1] == 'Status'] = 3
    dataset[:, 1][dataset[:, 1] == 'Video'] = 4

#get data set and change every range to number between 1-6
def likes_to_range(dataset):
    dataset[:, 16][dataset[:, 16] < 50] = 1
    dataset[:, 16][(dataset[:, 16] >= 50) & (dataset[:, 16] < 100)] = 2
    dataset[:, 16][(dataset[:, 16] >= 100) & (dataset[:, 16] < 200)] = 3
    dataset[:, 16][(dataset[:, 16] >= 200) & (dataset[:, 16] < 500)] = 4
    dataset[:, 16][dataset[:, 16] >= 500] = 5


def delete_columns(data):
    np.delete(data, 16, axis=1)
    # add all the rest


def pre_process_data(data):
    delete_empty_rows(data)
    delete_columns(data)

    # norm the type
    type_to_number(data)

    data = data.astype(np.int)

    # norm likes range
    likes_to_range(data)


def train_test_separate(data):

    train_x, train_y, test_x, test_y = None
    return train_x, train_y, test_x, test_y

def main():
    #for data - 16 is like
    filename = 'data\dataset_Facebook.csv'

    data = np.loadtxt(filename, dtype=str, delimiter=";")

    pre_process_data(data)

    np.savetxt('processedDataAll.csv', data, fmt='%s', delimiter=';')

    train_test_separate(data)

    #sava train and test




if __name__ == '__main__':
    main()