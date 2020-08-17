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

    return data


# change from names od Type to numbers
def type_to_number(dataset):
    # run on all data lines and change the second column - normalize between 1 to 4
    dataset[:, 1][dataset[:, 1] == 'Link'] = 1
    dataset[:, 1][dataset[:, 1] == 'Photo'] = 2
    dataset[:, 1][dataset[:, 1] == 'Status'] = 3
    dataset[:, 1][dataset[:, 1] == 'Video'] = 4

#get data set and change every range to number between 1-6
def likes_to_range(dataset):
    dataset[:, 7][dataset[:, 7] < 50] = 1
    dataset[:, 7][(dataset[:, 7] >= 50) & (dataset[:, 7] < 100)] = 2
    dataset[:, 7][(dataset[:, 7] >= 100) & (dataset[:, 7] < 200)] = 3
    dataset[:, 7][(dataset[:, 7] >= 200) & (dataset[:, 7] < 500)] = 4
    dataset[:, 7][dataset[:, 7] >= 500] = 5


def delete_columns(data):
    np.delete(data, 16, axis=1)
    # add all the rest


def pre_process_data(data):
    data = delete_empty_rows(data)

    # norm the type
    type_to_number(data)

    data = data.astype(np.int)

    # norm likes range
    likes_to_range(data)

    return data


def train_test_separate(data):

    # shuffle randim lines
    # np.random.shuffle(data)

    num_of_rows = data[:, 0].size

    # here 80 percent of train
    train_size = 0.8 * num_of_rows
    train_size = int(train_size)

    data_train = data[0:train_size]
    data_test = data[train_size:num_of_rows]

    # take the likes as targets
    train_y = data_train[:, 7]
    test_y = data_test[:, 7]

    # delete it from features
    train_x = np.delete(data_train, 7, axis=1)
    test_x = np.delete(data_test, 7, axis=1)

    return train_x, train_y, test_x, test_y

def main():
    #for data - 16 is like
    filename = 'data\dataset_Facebook.csv'

    #load text and ignore some coulmns
    data = np.loadtxt(filename, dtype=str, delimiter=";", usecols=(0,1,2,3,4,5,6,16))

    data_after_preprocess = pre_process_data(data)

    #np.savetxt('processedDataAll.csv', data, fmt='%s', delimiter=';')
    np.savetxt('processedDataAll_changed.csv', data_after_preprocess, fmt='%s', delimiter=';')

    train_x, train_y, test_x, test_y = train_test_separate(data_after_preprocess)

    np.savetxt('train_x.csv', train_x, fmt='%s', delimiter=';')
    np.savetxt('train_y.csv', train_y, fmt='%s', delimiter=';')
    np.savetxt('test_x.csv', test_x, fmt='%s', delimiter=';')
    np.savetxt('test_y.csv', test_y, fmt='%s', delimiter=';')



    #sava train and test




if __name__ == '__main__':
    main()