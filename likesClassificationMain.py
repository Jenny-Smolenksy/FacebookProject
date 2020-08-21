from decisionTreeFB import DecisionTreeFB
from logisticRegressionFB import LogisticRegressionFB
from neuralNetworkFB import NeuralNetworkFB
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import numpy as np


def draw_class(y_test, y_pred):
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    cnf_matrix

    class_names = [0, 1, 2, 3, 4]  # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    plt.show()


def main():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

    # preProcessing
    # load train set
    train_file = "data/train.csv"
    test_file = "data/test.csv"

    neural_net = NeuralNetworkFB(train_file)
    # train with best hyper parameters - may be taken from best params
    neural_net.train_model(100, 0.02, True, False, 0.005)
    # neural_net.cross_validation()
    # best_params_net = neural_net.check_hyper_parameters()
    neural_net.predict_test(test_file)  # all predictions

    decision_tree = DecisionTreeFB(train_file)
    # train with best
    decision_tree.train_tree("entropy", "random", 11, 0.01)
    # decision_tree.cross_validation()
    # best_params_tree = decision_tree.check_hyper_parameters()
    decision_tree.predict_test(test_file)

    logistic_regression = LogisticRegressionFB(train_file)
    # logistic_regression.cross_validation()
    # best_params = logistic_regression.check_hyper_parameters()
    logistic_regression.train_logistic_regression()
    # train with best
    logistic_regression.predict_test(test_file)

    data_x_test = np.loadtxt(test_file, skiprows=1, delimiter=';', usecols=range(0, 12))
    data_y_test = np.loadtxt(test_file, skiprows=1, delimiter=';', usecols=12)

    count_ensemble_correct = 0
    for i in range(len(data_x_test)):
        sample = data_x_test[i]
        real_tag = int(data_y_test[i])
        y_tag_net = neural_net.predict_sample_tag(sample)
        y_tag_tree = decision_tree.predict_sample(sample)
        y_tag_regression = logistic_regression.predict_sample(sample)
        print(f"real tag : {real_tag}, prediction net : {y_tag_net}, "
              f"prediction tree : {y_tag_tree}, prediction regression : {y_tag_regression}")

        tags = [y_tag_net, y_tag_tree, y_tag_regression]
        most_common = max(set(tags), key = tags.count)
        print(f"ensemble tag : {most_common}, real tag : {real_tag}")
        if most_common == real_tag:
            count_ensemble_correct += 1
    count_ensemble_correct /= len(data_y_test)
    count_ensemble_correct = round(count_ensemble_correct, 3)
    print(f"ensemble accuracy over test set: {count_ensemble_correct}")


if __name__ == '__main__':
    main()
