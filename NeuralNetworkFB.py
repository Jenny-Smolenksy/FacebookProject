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


class NeuralNet(nn.Module):
    """
    this class is the model for this assignment
    """

    def __init__(self):
        super(NeuralNet, self).__init__()

        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(12)
        self.linear1 = nn.Linear(12, 128)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.linear2 = nn.Linear(128, 64)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.linear3 = nn.Linear(64, 32)
        self.batch_norm3 = nn.BatchNorm1d(32)
        self.linear4 = nn.Linear(32, 5)
        self.linear5 = nn.Linear(5, 5)

    def forward(self, x):
        x = Variable(x)
        if torch.cuda.is_available():  # use cuda if possible
            x = x.cuda()

        x = self.batch_norm(x)
        x = self.relu(self.linear1(x))
        x = self.batch_norm1(x)
        x = self.relu(self.linear2(x))
        x = self.batch_norm2(x)
        x = self.relu(self.linear3(x))
        x = self.batch_norm3(x)
        x = self.relu(self.linear4(x))

        return x


def train(model, train_loader, learning_rate=0.001):
    """
    this function train the model
    :param model: to train
    :param train_loader: data to train by
    :param learning_rate:
    :return: none
    """
    loss_function = nn.CrossEntropyLoss()  # set loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # go over batches
    # for specs, classes in train_loader:
    for i, (inputs, targets) in enumerate(train_loader):

        inputs = Variable(inputs)
        targets = Variable(targets)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()

        optimizer.zero_grad()
        output = model(inputs)  # get prediction
        loss = loss_function(output, targets)  # calculate loss
        loss.backward()  # back propagation
        optimizer.step()  # optimizer step


def evaluate(data_loader, model):
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
        # output = torch.autograd.Variable(output, requires_grad=True)
        #  output = Variable(output)
        _, pred = torch.max(output.data, 1)  # get index of max log - probability
        correct += pred.eq(labels.view_as(pred)).cpu().sum()  # get correct in currect batch
    count_samples = len(data_loader.dataset)
    accuracy = (1. * correct / count_samples)
    return round(accuracy.item(), 3)


def cross_validation(data, model, num_of_epochs=100, learning_rate=0.01):
    torch.save(model, "model")
    train_acc_cross, valid_acc_cross = [], []

    from sklearn.model_selection import StratifiedKFold

    samples = data.specs
    tags = data.classes
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
    for train_index, valid_index in skf.split(samples, tags):
        print('k-cross validation')
        train_data = FBPostData(samples[train_index], tags[train_index])
        train_loader = \
            torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

        validation_data = FBPostData(samples[valid_index], tags[valid_index])
        validation_loader = \
            torch.utils.data.DataLoader(validation_data, batch_size=32, shuffle=True)

        model = torch.load("model")  # to start training from beginning
        train_acc, valid_acc, epochs = \
            run_epochs(model, train_loader, validation_loader, num_of_epochs, learning_rate)
        train_acc_cross.append(max(train_acc))
        valid_acc_cross.append(max(valid_acc))
        draw_graph_accuracy(epochs, train_acc, valid_acc)

    print(f"max train accuracy: {max(train_acc_cross)},"
          f" max validation accuracy: {max(valid_acc_cross)}")
    from statistics import mean
    print(f"average train accuracy: {mean(train_acc_cross)},"
          f" average validation accuracy: {mean(valid_acc_cross)}")


def run_epochs(model, train_loader, validation_loader, num_of_epochs=100, learning_rate=0.01):
    train_acc = []
    valid_acc = []
    counter_for_over_fitting = 0

    for epoch in range(num_of_epochs):
        print(f'epoch: {epoch + 1}')

        model.train()  # move to train mode
        train(model, train_loader, learning_rate)  # train

        model.eval()  # move to valuation mode
        train_accuracy = evaluate(train_loader, model)  # valuate
        train_acc.append(train_accuracy)
        print(f"train set accuracy: {train_accuracy}")

        valid_accuracy = evaluate(validation_loader, model)
        print(f"validation set accuracy: {valid_accuracy}")
        valid_acc.append(valid_accuracy)

        if train_accuracy - valid_accuracy > 0.05:
            counter_for_over_fitting += 1
            if counter_for_over_fitting > 5:
                break

    return train_acc, valid_acc, epoch+1  # if want to print


# def predict_test(classes, test_loader, model, file):
#     """
#     this function predicts on test
#     :param classes: options of tags
#     :param test_loader: data loader
#     :param model: to predict with
#     :param file: to write to
#     :return:
#     """
#     file_index = 0
#     names = test_loader.dataset.spects
#     for data, _ in test_loader:
#
#         output = model(data)  # get prediction
#         output = torch.autograd.Variable(output, requires_grad=True)
#         pred = output.max(1, keepdim=True)[1]  # get index of max log - probability
#
#         for element in pred:  # go over samples in the batch
#             label = classes[element.item()]  # get class name
#             file_name = ntpath.basename(names[file_index][0])  # get file name
#             if file:
#                 file.write(f"{file_name}, {label}\n")  # write to file
#             file_index += 1
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
    #plt.savefig('accuracy.png')


def main():
    """
    main function for ex5
    :return: 0 if successfully ended
    """

    train_file = "data/train.csv"
    data_x = np.loadtxt(train_file, skiprows=1, delimiter=';', usecols=range(0, 12))
    data_y = np.loadtxt(train_file, skiprows=1, delimiter=';', usecols=12)
    data_set = FBPostData(data_x, data_y)

    model = NeuralNet()
    if torch.cuda.is_available():
        model.cuda()

    cross_validation(data_set, model)

    model.eval()
    print('hello')


if __name__ == '__main__':
    main()
