from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import csv
import numpy as np

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
        self.model = None

    def cross_validation(self, c_value = 100000000000000000 , solver='lbfgs', max_iter=100):
        train_acc_cross, valid_acc_cross = [], []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
        data_features = np.array(self.data_features)
        data_tags = np.array(self.data_tags)

        for train_index, valid_index in skf.split(data_features, data_tags):
            train_data = data_features[train_index]
            train_tags = data_tags[train_index]

            validation_data = data_features[valid_index]
            validation_tags = data_tags[valid_index]

            logreg = LogisticRegression(penalty='l2', C=c_value, solver=solver, max_iter=max_iter,
                                    multi_class='multinomial')

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

    def check_hyper_parameters(self):
        c_value = [0.01, 0.1, 1, 10, 100, 1000, 10000]
        solver = ['newton-cg', 'lbfgs', 'sag', 'saga']
        max_iter = [100, 1000, 100000, 1000000]

        parms = []
        train = []
        valid = []
        all_combinations = [[j, k, z]
                            for j in c_value
                            for k in solver
                            for z in max_iter]

        for penalty, c_value, solver, max_iter in all_combinations:
            print(f"C: {c_value}, solver: {solver}, max_iter: {max_iter}")
            mean_train, mean_valid = self.cross_validation(self, c_value, solver, max_iter)
            parms.append([penalty, c_value, solver, max_iter])
            train.append(mean_train)
            valid.append(mean_valid)

        best_index = valid.index(max(valid))
        best_params = parms[best_index]
        print(f"best in: C={best_params[0]}, solver={best_params[1]},"
              f"max_iter={best_params[2]} \n"
              f" train accuracy={train[best_index]}, valid accuracy={valid[best_index]}")
        dict_best = {
            "c": best_params[0],
            "solver": best_params[1],
            "max_iter": best_params[2]}
        return dict_best

    def train_logistic_regression(self, c_value = 100000000000000000 , solver='lbfgs', max_iter=100):
        print('training logistic regression model')
        self.model = LogisticRegression(penalty='l2', C=c_value, solver=solver, max_iter=max_iter,
                                    multi_class='multinomial')
        self.model = self.model.fit(self.data_features, self.data_tags)


    def predict_test(self, test_data_file):
        data_x_test = np.loadtxt(test_data_file, skiprows=1, delimiter=';', usecols=range(0, 12))
        data_y_test = np.loadtxt(test_data_file, skiprows=1, delimiter=';', usecols=12)

        y_pred = self.model.predict(data_x_test)
        accuracy = metrics.accuracy_score(data_y_test, y_pred)
        accuracy = round(accuracy,3)
        print(f"accuracy on test set: {accuracy}")

        return accuracy

    def predict_sample(self, sample):
        sample = np.reshape(sample, (1, len(sample)))
        output = self.model.predict(sample)
        return output.item()



