from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier

#adaptive boosting

data, target = load_breast_cancer(return_X_y=True)

modelClf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=100, random_state=12)

X_train, X_valid, y_train, y_valid = train_test_split(data, target, test_size=0.3, random_state=12)

modelClf.fit(X_train, y_train)
print(modelClf.score(X_valid, y_valid))

#gradient boosting
from sklearn.ensemble import GradientBoostingClassifier

data, target = load_breast_cancer(return_X_y=True)

modelClf = GradientBoostingClassifier(max_depth=2, n_estimators=150,
                                      random_state=12, learning_rate=1)

X_train, X_valid, y_train, y_valid = train_test_split(data, target,
                                                      test_size=0.3, random_state=12)

modelClf.fit(X_train, y_train)
print(modelClf.score(X_valid, y_valid))