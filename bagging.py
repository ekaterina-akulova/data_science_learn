from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier

data, target = load_breast_cancer(return_X_y=True)

modelClf = BaggingClassifier(base_estimator=LogisticRegression(), n_estimators=50, random_state=12)

X_train, X_valid, y_train, y_valid = train_test_split(data, target, test_size=0.3, random_state=12)

modelClf.fit(X_train, y_train)
print(modelClf.score(X_valid, y_valid))