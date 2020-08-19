# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

def main():
    """
    main function for decision tree
    :return: 0 if successfully ended
    """

    train_file = "data/processedDataAll.csv"

    col_names = ['Page total likes','Type','Category','Post Month',
                 'Post Weekday','Post Hour','Paid','Lifetime Post Total Reach',
                 'Lifetime Post Total Impressions','Lifetime Post Impressions by people who have liked your Page',
                 'Lifetime Post reach by people who like your Page','comment','like']
    # load dataset
    pima = pd.read_csv(train_file,skiprows=1, header=None,delimiter=';', names=col_names)

    pima.head()

    # split dataset in features and target variable
    feature_cols = ['Page total likes','Type','Category','Post Month',
                 'Post Weekday','Post Hour','Paid','Lifetime Post Total Reach',
                 'Lifetime Post Total Impressions','Lifetime Post Impressions by people who have liked your Page',
                 'Lifetime Post reach by people who like your Page','comment']
    feature_col_y = ['like']
    X = pima[feature_cols]  # Features
    y = pima.like  # Target variable

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=1)  # 70% training and 30% test

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=10)

    # Train Decision Tree Classifer
    clf = clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


    print('hello')

if __name__ == '__main__':
    main()