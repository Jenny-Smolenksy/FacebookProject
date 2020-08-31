from decisionTreeFB import DecisionTreeFB
from logisticRegressionFB import LogisticRegressionFB
from neuralNetworkFB import NeuralNetworkFB
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import numpy as np
import dataPreProcessing


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
    import sys
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")

    # preProcessing
    # dataPreProcessing.data_pre_processing()

    # load train set
    train_file = "data/train.csv"
    test_file = "data/test.csv"

    neural_net = NeuralNetworkFB(train_file)
    decision_tree = DecisionTreeFB(train_file)
    logistic_regression = LogisticRegressionFB(train_file)

    # neural_net.train_model_and_check_validation(30, 0.01, False, False, 0, draw_accuracy=True)
    # neural_net.cross_validation(30, 0.01, l1_regular=False, l2_regular=False,
    #                             reg_labmda=0, draw_accuracy=False)  # without regularization

    #  neural_net.train_model_and_check_validation(30, 0.01, l1_reg=True, l2_reg=False,
    #                                             lambda_reg=0.005, draw_accuracy=True) # with l1 regularization
    # neural_net.cross_validation(30, 0.01, l1_regular=False, l2_regular=True,
    #                             reg_labmda=0.005)  # with l1 regularization

    # neural_net.train_model_and_check_validation(30, 0.01, l1_reg=False, l2_reg=True,
    #                                            lambda_reg=0.001, draw_accuracy=True)  # with l2 regularization
    # neural_net.cross_validation(30, 0.01, l1_regular=False, l2_regular=True,
    #                             reg_labmda=0.001)  # with l2 regularization

    # best_params_net = neural_net.check_hyper_parameters()

    # decision_tree.train_by_samples(criterion="entropy", splitter="best",
    #                                          max_depth=10, ccp_alpha=0)  # without regularization
    # decision_tree.cross_validation(criterion="entropy", splitter="best", max_depth=10, ccp_alpha=0)
    # decision_tree.train_by_samples(criterion="entropy", splitter="best", max_depth=10,
    #                                           ccp_alpha=0.05)  # with regularization
    #
    # decision_tree.cross_validation(criterion="entropy", splitter="best", max_depth=10, ccp_alpha=0.05)
    # best_params_tree = decision_tree.check_hyper_parameters()

    # logistic_regression.train_by_samples(c_value=100000000000000000, solver='newton-cg',
    #                                    max_iter=1000)  # without regularization
    # logistic_regression.cross_validation(c_value=100000000000000000, solver='lbfgs',
    #                                                  max_iter=1000)
    #
    # logistic_regression.train_by_samples(c_value=1000, solver='newton-cg', max_iter=1000)  # with regularization
    #
    # logistic_regression.cross_validation(c_value=100, solver='lbfgs', max_iter=1000)
    # best_params = logistic_regression.check_hyper_parameters()

    neural_net.train_model(epochs=50, lr=0.05, l1_reg=False, l2_reg=True, lambda_reg=0.1)
    neural_net.predict_test(test_file)

    decision_tree.train_tree(criterion="gini", splitter="random", max_depth=11, ccp_alpha=0.002)
    # decision_tree.tree_visualization()
    decision_tree.predict_test(test_file)

    logistic_regression.train_logistic_regression(c_value=1, solver='newton-cg', max_iter=1000)
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
        most_common = max(set(tags), key=tags.count)
        print(f"ensemble tag : {most_common}, real tag : {real_tag}")
        if most_common == real_tag:
            count_ensemble_correct += 1

    count_ensemble_correct /= len(data_y_test)
    count_ensemble_correct = round(count_ensemble_correct, 3)
    print(f"ensemble accuracy over test set: {count_ensemble_correct}")


if __name__ == '__main__':
    main()
