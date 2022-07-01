import  sklearn
from sklearn.ensemble import StackingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

data, target = load_breast_cancer(return_X_y=True)

estimators = [('lr', LogisticRegression()), ('dt', DecisionTreeClassifier())]
modelClf = StackingClassifier(estimators=estimators, final_estimator=SVC())

X_train, X_valid, y_train, y_valid = train_test_split(data, target, test_size=0.3, random_state=12)

modelClf.fit(X_train, y_train)
print(modelClf.score(X_valid, y_valid))