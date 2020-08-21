import numpy as np
from scipy import stats


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
    dataset[:, 12][dataset[:, 12] < 50] = 0
    dataset[:, 12][(dataset[:, 12] >= 50) & (dataset[:, 12] < 100)] = 1
    dataset[:, 12][(dataset[:, 12] >= 100) & (dataset[:, 12] < 200)] = 2
    dataset[:, 12][(dataset[:, 12] >= 200) & (dataset[:, 12] < 500)] = 3
    dataset[:, 12][dataset[:, 12] >= 500] = 4

def delete_columns(data):
    np.delete(data, 12, axis=1)
    # add all the rest


def pre_process_data(data):

    title_row = data[0]
    data = delete_empty_rows(data)

    # norm the type
    type_to_number(data)

    data = data.astype(np.int)

    # norm likes range
    likes_to_range(data)

    return data, title_row

def normalization_data_divide_by_max(data):
    data_norm = data / (data.max(axis=0) + np.spacing(0))
    return data_norm

def norm_min_max(data):
    data_norm  = (data - data.min()) / (data.max() - data.min())
    return data_norm

def norm_standart(data):
    data_norm = (data - data.mean()) / data.std()
    return data_norm

def norm_zcore(data):
    stats.zscore(data)

def train_test_separate(data):

    # shuffle randim lines
    np.random.shuffle(data)

    num_of_rows = data[:, 0].size

    # here 80 percent of train
    train_size = 0.8 * num_of_rows
    train_size = int(train_size)

    data_train = data[0:train_size]
    data_test = data[train_size:num_of_rows]


    # take the likes as targets
    train_y = data_train[:, 12]
    test_y = data_test[:, 12]

    # delete it from features
    train_x = np.delete(data_train, 12 , axis=1)
    test_x = np.delete(data_test, 12, axis=1)

    return train_x, train_y, test_x, test_y,data_train,data_test

def data_pre_processing():
    #for data - 16 is like
    filename = 'data\dataset_Facebook.csv'

    #load text and ignore some coulmns
    data = np.loadtxt(filename, dtype=str, delimiter=";", usecols=(0,1,2,3,4,5,6,7,8,12,13,17,16))

    data_after_precess, title_row = pre_process_data(data)

    data_with_title = np.vstack((title_row, data_after_precess))
    np.savetxt('data\processedDataAll.csv', data_with_title, fmt='%s', delimiter=';')

    train_x, train_y, test_x, test_y, data_train,data_test = train_test_separate(data_after_precess)

    train_with_title = np.vstack((title_row, data_train))
    np.savetxt('data/train.csv', train_with_title, fmt='%s', delimiter=';')
    test_with_title = np.vstack((title_row, data_test))
    np.savetxt('data/test.csv', test_with_title, fmt='%s', delimiter=';')


