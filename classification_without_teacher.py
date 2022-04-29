import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
        self.eta = eta #Темп обучения
        self.n_iter = n_iter #Количество итераций (уроков)

    def fit(self, X, y):
        self.w = np.zeros(1 + X.shape[1]) # w_ - одномерный массив - веса после обучения
        self.errors = [] # список ошибок
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
#  выборка из df 100 строк (столбец 0 и столбец 2), загрузка их в массив х
X = df.iloc[0:100, [0, 2]].values
print('Значение X - 100')
print(X)
#  выборка из df 100 строк (столбец 4) в массив чисел -1 и 1
y = df.iloc[0:100, 4].values
# преобразование названий цветков (столбец 4) в массив чисел -1 и 1
y = np.where(y == 'Iris-setosa', 1, 1)
print('Значение названий цветков в виде -1 и 1б Y - 100')
print(y)
#первые 50 элементов обучающей выборки (строки 0-5, столбцы 0,1)
plt.scatter(X[0:50, 0], X[0:50, 1], color='red', marker='o', label='щетинистый')
#Следующие 50 элементов обучающей выборки (строки 50-100, столбцы 0, 1)
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='разноцветный')
#формирование названий осей и вывод графика на экран
plt.xlabel('Длина чашелистника')
plt.ylabel('Длина лепестка')
plt.legend(loc='upper left')
plt.show()


