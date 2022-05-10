from sklearn import svm, datasets
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

# импортируем набор данных (например, возьмём тот же iris)
iris = datasets.load_iris()
X = iris.data[:, :2]  # возьмём только первые 2 признака, чтобы проще воспринять вывод
y = iris.target

C = 1.0  # параметр регуляризации SVM
svc = svm.SVC(kernel='linear', C=1, gamma=1).fit(X, y)  # здесь мы взяли линейный kernel

# создаём сетку для построения графика
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max),
                     np.arange(y_min, y_max))

plt.subplot(1, 1, 1)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Sepal length')  # оси и название укажем на английском
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')
plt.show()

