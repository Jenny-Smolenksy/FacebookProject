# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
import numpy as np

def draw_tree(clf,feature_cols):
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=feature_cols, class_names=['0', '1','2','3','4'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('facebook_dt.png')
    Image(graph.create_png())


def check_hyper_parameters(data_features, data_tags):

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
        mean_train, mean_valid = cross_validation(data_features, data_tags, criterion, splitter, max_depth, ccp_alpha)
        parms.append([criterion, splitter, max_depth, ccp_alpha])
        train.append(mean_train)
        valid.append(mean_valid)

    best_index = valid.index(max(valid))
    best_params = parms[best_index]
    print(f"best in: criterion={best_params[0]}, splitter={best_params[1]}, max_depth={best_params[2]},"
          f"ccp_alpha={best_params[3]} \n"
          f" train accuracy={train[best_index]}, valid accuracy={valid[best_index]}")


def cross_validation(data_features, data_tags, criterion, splitter, max_depth, ccp_alpha):

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

        clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, splitter=splitter, ccp_alpha=ccp_alpha)

        # Train Decision Tree Classifer
        clf = clf.fit(train_data, train_tags)

        # Predict the response for test dataset
        validation_predictions = clf.predict(validation_data)
        train_predictions = clf.predict(train_data)

        train_acc_cross.append(metrics.accuracy_score(train_tags, train_predictions))
        valid_acc_cross.append(metrics.accuracy_score(validation_tags, validation_predictions))


    print(f"max train accuracy: {round((max(train_acc_cross)).item(), 3)},"
          f" max validation accuracy: {round((max(valid_acc_cross)).item(), 3)}")
    from statistics import mean
    train_mean = round((mean(train_acc_cross)).item(),3)
    valid_mean = round((mean(valid_acc_cross)).item(),3)
    print(f"average train accuracy: {train_mean},"
           f" average validation accuracy: {valid_mean}")
    return train_mean, valid_mean


def main():
    """
    main function for decision tree
    :return: 0 if successfully ended
    """

    train_file = "data/processedDataAll.csv"

    col_names = ['Page total likes','Type','Category','Post Month',
                 'Post Weekday','Post Hour','Paid','Lifetime Post Total Reach',
                 'Lifetime Post Total Impressions','Lifetime Post Impressions by people who have liked your Page',
                 'Lifetime Post reach by people who like your Page','share','like']
    # load dataset
    pima = pd.read_csv(train_file,skiprows=1, header=None,delimiter=';', names=col_names)

    pima.head()

    # split dataset in features and target variable
    feature_cols = ['Page total likes','Type','Category','Post Month',
                 'Post Weekday','Post Hour','Paid','Lifetime Post Total Reach',
                 'Lifetime Post Total Impressions','Lifetime Post Impressions by people who have liked your Page',
                 'Lifetime Post reach by people who like your Page','share']
    feature_col_y = ['like']
    X = pima[feature_cols]  # Features
    y = pima.like  # Target variable

    check_hyper_parameters(X, y)

    # need to change to take the best from k
    clf = DecisionTreeClassifier(criterion='gini', max_depth=11, splitter='random', ccp_alpha=0.005)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=1)  # 70% training and 30% test

    clf = clf.fit(X_train, y_train)

    draw_tree(clf, feature_cols)

    #cross_validation(X,y)

    # # Split dataset into training set and test set
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
    #                                                     random_state=1)  # 70% training and 30% test
    #
    #
    # # Create Decision Tree classifer object
    # clf = DecisionTreeClassifier(criterion="entropy", max_depth=10)
    # X_train = np.array(X_train)
    # # Train Decision Tree Classifer
    # clf = clf.fit(X_train, y_train)
    #
    # # Predict the response for test dataset
    # y_pred = clf.predict(X_test)
    #
    # # Model Accuracy, how often is the classifier correct?
    # print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    #

    print('hello')

if __name__ == '__main__':
    main()