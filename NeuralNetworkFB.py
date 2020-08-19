
import torch.cuda
import torch.utils.data
import torch.cuda
from torch import nn, optim
from torch.autograd import Variable
from matplotlib import pyplot as plt
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
        self.linear1 = nn.Linear(12, 64)
        self.linear2 = nn.Linear(64, 128)
        self.batch_norm128 = nn.BatchNorm1d(128)
        self.linear3 = nn.Linear(128, 32)
        self.batch_norm32 = nn.BatchNorm1d(32)
        self.linear4 = nn.Linear(32, 5)

    def forward(self, x):

        x = Variable(x)
        if torch.cuda.is_available():  # use cuda if possible
            x = x.cuda()

        x = self.batch_norm(x)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.batch_norm128(x)
        x = self.relu(self.linear3(x))
        x = self.batch_norm32(x)
        x = self.relu(self.linear4(x))

        return x



def train(model, train_loader, learning_rate=0.001, l2_regular=False, l1_regular=False, reg_labmda=0.01):
    """
    this function train the model
    :param model: to train
    :param train_loader: data to train by
    :param learning_rate:
    :return: none
    """
    loss_function = nn.CrossEntropyLoss()  # set loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if l2_regular:
        optimizer.weight_decay = reg_labmda

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
        if l1_regular:
            reg = 0
            for params in model.parameters():
                reg += abs(0.5*(params**2)).sum()
                loss += reg_labmda * reg


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


def cross_validation(data, model, num_of_epochs=100, learning_rate=0.01,
                     l2_regular=False, l1_regular=False, reg_labmda=0.01):
    torch.save(model, "model")
    train_acc_cross, valid_acc_cross = [], []

    from sklearn.model_selection import StratifiedKFold

    samples = data.specs
    tags = data.classes
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
    for train_index, valid_index in skf.split(samples, tags):
        train_data = FBPostData(samples[train_index], tags[train_index])
        train_loader = \
            torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

        validation_data = FBPostData(samples[valid_index], tags[valid_index])
        validation_loader = \
            torch.utils.data.DataLoader(validation_data, batch_size=32, shuffle=True)

        model = torch.load("model")  # to start training from beginning
        train_acc, valid_acc, epochs = \
            run_epochs(model, train_loader, validation_loader, num_of_epochs,
                       learning_rate, l2_regular, l1_regular, reg_labmda)
        train_acc_cross.append(max(train_acc))
        valid_acc_cross.append(max(valid_acc))


        #draw_graph_accuracy(epochs, train_acc, valid_acc)

    print(f"max train accuracy: {round((max(train_acc_cross)).item(),3)},"
          f" max validation accuracy: {round((max(valid_acc_cross)).item(),3)}")
    from statistics import mean
    train_mean = round((mean(train_acc_cross)).item(),3)
    valid_mean = round((mean(valid_acc_cross)).item(),3)
    print(f"average train accuracy: {train_mean},"
          f" average validation accuracy: {valid_mean}")
    return train_mean, valid_mean


def run_epochs(model, train_loader, validation_loader, num_of_epochs=100,
               learning_rate=0.01, l2_regular=True, l1_regular=False, reg_labmda=0.01):
    train_acc = []
    valid_acc = []
    counter_for_over_fitting = 0

    for epoch in range(num_of_epochs):
        #print(f'epoch: {epoch + 1}')

        model.train()  # move to train mode
        train(model, train_loader, learning_rate, l2_regular, l1_regular, reg_labmda)  # train

        model.eval()  # move to valuation mode
        train_accuracy = evaluate(train_loader, model)  # valuate
        train_acc.append(train_accuracy)
        #print(f"train set accuracy: {train_accuracy}")

        valid_accuracy = evaluate(validation_loader, model)
        #print(f"validation set accuracy: {valid_accuracy}")
        valid_acc.append(valid_accuracy)

        # if train_accuracy - valid_accuracy > 0.05:
        #     counter_for_over_fitting += 1
        #     if counter_for_over_fitting > 5:
        #         break

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


def check_hyper_parameters(model, data):
    learn_rate = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    lamdba_reg = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    epochs = [10, 20, 30, 50, 100]
    parms = []
    train = []
    valid = []
    all_combinations = [[i, j, k] for i in learn_rate
                 for j in lamdba_reg
                 for k in epochs]

    for lr, lamdba_reg, epochs in all_combinations:
        print(f"lr: {lr}, lamda: {lamdba_reg}, epochs: {epochs}")
        mean_train, mean_valid = cross_validation(data, model, epochs, lr, True, False, lamdba_reg)
        parms.append([lr,lamdba_reg,epochs])
        train.append(mean_train)
        valid.append(mean_valid)

    best_index = valid.index(max(valid))
    best_params = parms[best_index]
    print(f"best in: lr={best_params[0]}, lambda={best_params[1]}, epochs={best_params[2]} \n"
           f" train accuracy={train[best_index]}, valid accuracy={valid[best_index]}")


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

    #check_hyper_parameters(model, data_set)
    cross_validation(data_set, model, 100, 0.02, True, False, 0.005)
    #model.eval()
    #print('hello')

if __name__ == '__main__':
    main()
