import pandas
import torch.cuda
import torch.utils.data
import torch.cuda
from torch import nn, optim
from torch.autograd import Variable
from matplotlib import pyplot as plt
import ntpath
import numpy as np

from FBPostData import FBPostData


def train(model, train_loader, optimizer):
    """
    this function train the model
    :param model: to train
    :param train_loader: data to train by
    :param optimizer:
    :return: none
    """
    loss_function = nn.CrossEntropyLoss()  # set loss function

    # go over batches
    #for specs, classes in train_loader:
    for i, (inputs, targets) in enumerate(train_loader):
       # data = Variable(data)
      #  labels = Variable(labels)
      #   if torch.cuda.is_available():
      #       specs = specs.cuda()
      #       classes = classes.cuda()

        optimizer.zero_grad()

        output = model(inputs)  # get prediction



        loss = loss_function(output, targets)  # calculate loss
        loss.backward()  # back propagation
        optimizer.step()  # optimizer step

def run_epochs(epochs_num, model, train_loader, optimizer, validation_loader):
    """
    this function runs epochs of train and accuracy check
    :param epochs_num: to run
    :param model: to train
    :param train_loader: data to train by
    :param optimizer:
    :param validation_loader:
    :return: list of train and valid accuracy per epoch
    """

    train_acc = []
    valid_acc = []
    for epoch in range(epochs_num):
        print(f'epoch: {epoch + 1}')

        model.train()  # move to train mode
        train(model, train_loader, optimizer)  # train

        model.eval()  # move to valuation mode

        accuracy = valuation(train_loader, model)  # valuate
        train_acc.append(accuracy)
        print(f"train set accuracy: {accuracy}")
        accuracy = valuation(validation_loader, model)
        print(f"validation set accuracy: {accuracy}")
        valid_acc.append(accuracy)

    return train_acc, valid_acc

class NeuralNet(nn.Module):
    """
    this class is the model for this assignment
    """

    def __init__(self):
        super(NeuralNet, self).__init__()

        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(7, 128)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.linear2 = nn.Linear(128, 64)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.linear3 = nn.Linear(64, 5)


    def forward(self, x):
        x = Variable(x)
        if torch.cuda.is_available():  # use cuda if possible
            x = x.cuda()

        x = self.relu(self.linear1(x))
        x = self.batch_norm1(x)
        x = self.relu(self.linear2(x))
        x = self.batch_norm2(x)
        x = self.relu(self.linear3(x))

        return x





def valuation(data_loader, model):
    """
    this function checks accuracy, valuate predictions from real tags
    :param data_loader: data
    :param model: model to predict with
    :return: accuracy percentage
    """
    correct = 0
    # go over data
    for (data, labels) in data_loader:
        labels = Variable(labels)
        if torch.cuda.is_available():
            labels = labels.cuda()  # use cuda if possible

        output = model(data)  # get prediction
        output = torch.autograd.Variable(output, requires_grad=True)
        pred = output.max(1, keepdim=True)[1]  # get index of max log - probability
        correct += pred.eq(labels.view_as(pred)).cpu().sum()  # get correct in currect batch
    count_samples = len(data_loader.dataset)
    accuracy = (1. * correct / count_samples)
    return round(accuracy.item(), 3)


def predict_test(classes, test_loader, model, file):
    """
    this function predicts on test
    :param classes: options of tags
    :param test_loader: data loader
    :param model: to predict with
    :param file: to write to
    :return:
    """
    file_index = 0
    names = test_loader.dataset.spects
    for data, _ in test_loader:

        output = model(data)  # get prediction
        output = torch.autograd.Variable(output, requires_grad=True)
        pred = output.max(1, keepdim=True)[1]  # get index of max log - probability

        for element in pred:  # go over samples in the batch
            label = classes[element.item()]  # get class name
            file_name = ntpath.basename(names[file_index][0])   # get file name
            if file:
                file.write(f"{file_name}, {label}\n")  # write to file
            file_index += 1


def draw_graph_accuracy(num_of_epochs, train_accuracy, validation_accuracy):
    """
    this function creates graph of train and validation accuracy per epoch
    :param num_of_epochs:
    :param train_accuracy:
    :param validation_accuracy:
    :return: none
    """
    x = list(range(1, num_of_epochs + 1))
    plt.plot(x, validation_accuracy, label="validation", color='blue', linewidth=2)
    plt.plot(x, train_accuracy, label="train", color='green', linewidth=2)
    plt.title('Accuracy per epoch', fontweight='bold', fontsize=13)
    plt.xlabel('number of epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
    plt.savefig('accuracy.png')


def main():
    """
    main function for ex5
    :return: 0 if successfully ended
    """


    train_file = "data/train.csv"
    data_x = np.loadtxt(train_file, skiprows=1, delimiter=';', usecols=range(0,7))
    data_y = np.loadtxt(train_file, skiprows=1, delimiter=';', usecols=7)

    data_x = data_x.astype('float32')
    data_y = data_y.astype('float32')
    data_y = data_y.reshape((len(data_y), 1))

    data_set_train = FBPostData(data_x, data_y)

    train_loader = torch.utils.data.DataLoader(data_set_train, batch_size=32, shuffle=True)


    model = NeuralNet()

    if torch.cuda.is_available():
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    num_of_epochs = 5
    train_acc, validation_acc = run_epochs(num_of_epochs, model, train_loader, optimizer, validation_loader=train_loader)
    model.eval()
    print('hello')



if __name__ == '__main__':
    main()
