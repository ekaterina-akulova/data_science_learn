import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

#Описание класса


'''
Выполнить подгонку под тренировочные данные
Параметры
Х-тренирровочные двнные: массив, размерность - X[n_samples, n_features]
у_samples - число образцов
n_features - число признаков
y - целевые значения: массив, размерность -  y[n_samples]
'''


class Persiptron(object):
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta  #Темп обучения
        self.n_iter = n_iter  #Количество итераций (уроков)

    def fit(self, X, y): #веса будут обновляться путем минимизации функции стоимости методом градиентного спуска
        self.w_ = np.zeros(1 + X.shape[1])  # w_ - одномерный массив - веса после обучения

        self.errors_ = []  # список ошибок

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    '''Рассчитать чистый вход'''
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    '''Вернуть метку класса после единичного скачка'''
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)
print('Данные об ирисах')
print(df.to_string())
df.to_csv('C:\CSV\Iris.csv')
X = df.iloc[0:100, [0, 2]].values #  выборка из df 100 строк (столбец 0 и столбец 2), загрузка их в массив х
print('Значение X - 100')
print(X)
y = df.iloc[0:100, 4].values #  выборка из df 100 строк (столбец 4) в массив чисел -1 и 1
y = np.where(y == 'Iris-setosa', 1, 1) # преобразование названий цветков (столбец 4) в массив чисел -1 и 1
print('Значение названий цветков в виде -1 и 1б Y - 100')
print(y)
plt.scatter(X[0:50, 0], X[0:50, 1], color='red', marker='o', label='щетинистый') #первые 50 элементов обучающей выборки (строки 0-5, столбцы 0,1)
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='разноцветный') #Следующие 50 элементов обучающей выборки (строки 50-100, столбцы 0, 1)
plt.xlabel('Длина чашелистника') #формирование названий осей и вывод графика на экран
plt.ylabel('Длина лепестка')
plt.legend(loc='upper left')
plt.show()

#тренирока
ppn = Persiptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Эпохи')
plt.ylabel('Число случаев ошибочной классификации')
plt.show()

i1 = [5.5, 1.6]
i2 = [6.4, 4.5]
R1 = ppn.predict(i1)
R2 = ppn.predict(i2)
print('R1=', R1, ' R2=', R2)
if (R1==1):
    print('R1= Вид iris senosa')
else:
    print('R1= Вид iris versicolor')

#Визуализация разделительной границы
def plot_desicion_regions(X, y, classifier, resolution=0.02): #
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'green', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])


    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max),
        np.arange(x2_min, x2_max))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha = 0.4, cmap = map)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    #Показать образцы классов
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8,
                    c=cmap(idx), marker=markers[idx], label=cl)


plot_desicion_regions(X, y, classifier=ppn)
plt.xlabel('Длина чашелистика, см')
plt.ylabel('Длина лепестка, см')
plt.legend(loc='upper_left')
plt.show()


