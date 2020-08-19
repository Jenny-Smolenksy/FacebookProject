from sklearn.linear_model import LogisticRegression
import pandas
from sklearn.model_selection import train_test_split

logreg = LogisticRegression()

train_file = "data/processedDataAll.csv"
# load dataset
col_names = ['Page total likes', 'Type', 'Category', 'Post Month',
             'Post Weekday', 'Post Hour', 'Paid', 'Lifetime Post Total Reach',
             'Lifetime Post Total Impressions', 'Lifetime Post Impressions by people who have liked your Page',
             'Lifetime Post reach by people who like your Page', 'comment', 'like']


pima = pandas.read_csv(train_file, skiprows=1, header=None, delimiter=';', names=col_names)

pima.head()

# split dataset in features and target variable
feature_cols = ['Page total likes', 'Type', 'Category', 'Post Month',
                'Post Weekday', 'Post Hour', 'Paid', 'Lifetime Post Total Reach',
                'Lifetime Post Total Impressions', 'Lifetime Post Impressions by people who have liked your Page',
                'Lifetime Post reach by people who like your Page', 'comment']
feature_col_y = ['like']
X = pima[feature_cols]  # Features
y = pima.like  # Target variable

# Split dataset into training set and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=1)  # 70% training and 30% test

logreg.fit(X_train,y_train)

#
y_pred=logreg.predict(X_test)

from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))