import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data[:, np.newaxis, 2]
x_train = diabetes_X[:-20] #split the data into training set
x_test = diabetes_X[-20:] #split the data into testing set
y_train = diabetes.target[:-20] #split the target into training set
y_test = diabetes.target[-20:] #split the target into testing set

#creating and teaching model
lr = LinearRegression()
lr.fit(x_train, y_train)
predictions = lr.predict(x_test)

#the mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, predictions))
#explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, predictions))

plt.scatter(x_test, y_test, color='black')
plt.plot(x_test, predictions, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()