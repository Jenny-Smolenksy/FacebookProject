from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import numpy as np
import csv


class LogisticRegressionFB:

    def __init__(self, data_file):
        with open(data_file, 'r') as infile:
            reader = csv.DictReader(infile)
            fieldnames = reader.fieldnames
        col_names = fieldnames[0].split(';')
        # load dataset
        pima = pd.read_csv(data_file, skiprows=1, header=None, delimiter=';', names=col_names)
        pima.head()
        # split dataset in features and target variable
        feature_cols = col_names[:-1]
        self.data_features = pima[feature_cols]  # Features
        self.data_tags = pima.like  # Target variable


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


def check_hyper_parameters(data_features, data_tags):
    penalty = ['l2', 'elasticnet', 'none']
    c_value = [2, 1, 0.5, 0.1, 0.001, 0.0001, 0.02, 0.005, 0.0005]
    solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    max_iter = [10000, 20000, 50000, 100000]

    parms = []
    train = []
    valid = []
    all_combinations = [[i, j, k, z] for i in penalty
                        for j in c_value
                        for k in solver
                        for z in max_iter]

    for penalty, c_value, solver, max_iter in all_combinations:
        print(f"penately: {penalty}, C: {c_value}, solver: {solver}, max_iter: {max_iter}")
        mean_train, mean_valid = cross_validation(data_features, data_tags, penalty, c_value, solver, max_iter)
        parms.append([penalty, c_value, solver, max_iter])
        train.append(mean_train)
        valid.append(mean_valid)

    best_index = valid.index(max(valid))
    best_params = parms[best_index]
    print(f"best in: penately={best_params[0]}, C={best_params[1]}, solver={best_params[2]},"
          f"max_iter={best_params[3]} \n"
          f" train accuracy={train[best_index]}, valid accuracy={valid[best_index]}")


def cross_validation(data_features, data_tags, penalty, c_value, solver, max_iter):
    train_acc_cross, valid_acc_cross = [], []

    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)

    data_features = np.array(data_features)
    data_tags = np.array(data_tags)

    for train_index, valid_index in skf.split(data_features, data_tags):
        train_data = data_features[train_index]
        train_tags = data_tags[train_index]

        validation_data = data_features[valid_index]
        validation_tags = data_tags[valid_index]

        logreg = LogisticRegression(penalty=penalty, C=c_value, solver=solver, max_iter=max_iter)

        # Train Logistic regression Tree Classifer
        logreg = logreg.fit(train_data, train_tags)

        # Predict the response for test dataset
        validation_predictions = logreg.predict(validation_data)
        train_predictions = logreg.predict(train_data)

        train_acc_cross.append(metrics.accuracy_score(train_tags, train_predictions))
        valid_acc_cross.append(metrics.accuracy_score(validation_tags, validation_predictions))

    print(f"max train accuracy: {round((max(train_acc_cross)).item(), 3)},"
          f" max validation accuracy: {round((max(valid_acc_cross)).item(), 3)}")
    from statistics import mean
    train_mean = round((mean(train_acc_cross)).item(), 3)
    valid_mean = round((mean(valid_acc_cross)).item(), 3)
    print(f"average train accuracy: {train_mean},"
          f" average validation accuracy: {valid_mean}")
    return train_mean, valid_mean


def main():
    # check
    check_hyper_parameters(X, y)

    #  logreg = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
    #                              intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear',
    #                              max_iter=10000, multi_class='ovr', verbose=0)
    # Split dataset into training set and test set

    print("hello")


if __name__ == '__main__':
    main()
