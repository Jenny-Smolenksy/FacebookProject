# Load libraries
from io import StringIO

import pandas as pd
import pydotplus
from sklearn.tree import DecisionTreeClassifier, export_graphviz  # Import Decision Tree Classifier
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn.model_selection import StratifiedKFold
import numpy as np
import csv
from IPython.display import Image


class DecisionTreeFB:
    def __init__(self, train_file):

        with open(train_file, 'r') as infile:
            reader = csv.DictReader(infile)
            fieldnames = reader.fieldnames
        col_names = fieldnames[0].split(';')
        # load dataset
        pima = pd.read_csv(train_file, skiprows=1, header=None, delimiter=';', names=col_names)
        pima.head()
        # split dataset in features and target variable
        self.feature_cols = col_names[:-1]
        self.data_features = pima[self.feature_cols]  # Features
        self.data_tags = pima.like  # Target variable
        self.clf = None

    def create_tree(self, criterion="entropy", splitter="best", max_depth=10, ccp_alpha=0.05):
        self.clf = DecisionTreeClassifier(criterion="entropy", max_depth=10)
        clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, splitter=splitter, ccp_alpha=ccp_alpha)

    def cross_validation(self, criterion="entropy", splitter="best", max_depth=10, ccp_alpha=0.05):

        train_acc_cross, valid_acc_cross = [], []

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
        data_features = np.array(self.data_features)
        data_tags = np.array(self.data_tags)

        for train_index, valid_index in skf.split(data_features, data_tags):
            train_data = data_features[train_index]
            train_tags = data_tags[train_index]

            validation_data = data_features[valid_index]
            validation_tags = data_tags[valid_index]

            self.create_tree(criterion, splitter, max_depth, ccp_alpha)

            # Train Decision Tree Classifer
            self.clf = self.clf.fit(train_data, train_tags)

            # Predict the response for test dataset
            validation_predictions = self.clf.predict(validation_data)
            train_predictions = self.clf.predict(train_data)

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

    def check_hyper_parameters(self):

        criterion = ["gini", "entropy"]
        splitter = ["best", "random"]
        max_depth = [7, 8, 9, 10, 11, 12]
        ccp_alphas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

        parms = []
        train = []
        valid = []
        all_combinations = [[i, j, k, z] for i in criterion
                            for j in splitter
                            for k in max_depth
                            for z in ccp_alphas]

        for criterion, splitter, max_depth, ccp_alpha in all_combinations:
            print(f"criterion: {criterion}, splitter: {splitter}, max_depth: {max_depth}, ccp_alpha: {ccp_alpha}")
            mean_train, mean_valid = self.cross_validation(criterion, splitter, max_depth, ccp_alpha)
            parms.append([criterion, splitter, max_depth, ccp_alpha])
            train.append(mean_train)
            valid.append(mean_valid)

        best_index = valid.index(max(valid))
        best_params = parms[best_index]
        print(f"best in: criterion={best_params[0]}, splitter={best_params[1]}, max_depth={best_params[2]},"
              f"ccp_alpha={best_params[3]} \n"
              f" train accuracy={train[best_index]}, valid accuracy={valid[best_index]}")
        dict_best = {
            "criterion": best_params[0],
            "splitter": best_params[1],
            "max_depth": best_params[2],
            "ccp_alpha": best_params[3]}
        return dict_best

    def train_tree(self, criterion="entropy", splitter="best", max_depth=10, ccp_alpha=0.05):
        print('training decision tree model')
        self.create_tree(criterion, splitter, max_depth, ccp_alpha)
        # train using all train data
        self.clf = self.clf.fit(self.data_features, self.data_tags)

    def predict_test(self, test_data_file):
        data_x_test = np.loadtxt(test_data_file, skiprows=1, delimiter=';', usecols=range(0, 12))
        data_y_test = np.loadtxt(test_data_file, skiprows=1, delimiter=';', usecols=12)

        y_pred = self.clf.predict(data_x_test)
        accuracy = metrics.accuracy_score(data_y_test, y_pred)
        accuracy = round(accuracy, 3)
        print(f"accuracy on test set: {accuracy}")

        return accuracy

    def predict_sample(self, sample):
        sample = np.reshape(sample, (1, len(sample)))
        output = self.clf.predict(sample)
        return output.item()

    def tree_visualization(self):
        dot_data = StringIO()
        export_graphviz(self.clf, out_file=dot_data,
                        filled=True, rounded=True,
                        special_characters=True, feature_names=self.feature_cols, class_names=['0', '1', '2', '3', '4'])
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_png('decision_tree.png')
        #Image(graph.create_png())
