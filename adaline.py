import datasets as datasets
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_decision_regions


class   AdaptiveLinearNeuron(object):
    def __init__(self, rate=0.01, niter=10):
        self.rate = rate # rate - темп обучения (между 0.0 и 1.0)
        self.niter = niter # niter - проходы по тренировочному набору данных

    def fit(self, X, y):
        self.weight = np.zeros(1 + X.shape[1])
        self.cost = []
        for i in range(self.niter):
            output = self.net_input(X)
            errors = y - output
            self.weight[1:] += self.rate * X.T.dot(errors)
            self.weight[0] += self.rate * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost.append(cost)
        return self

    def net_input(self, X): # вычисление чистого входного сигнала
            return np.dot(X, self.weight[1:]) + self.weight[0]

    def activation(self, X): # вычисление линейной активации
            return self.net_input(X)

    def predict(self, X): #возвращает метку класса после единичного шага(предсказание)
        return np.where(self.activation(X) >= 0.0, 1, -1)


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)
print('Данные об ирисах')
print(df.to_string())
df.to_csv('C:\CSV\Iris.csv')
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
a1n1 = AdaptiveLinearNeuron(0.01, 10).fit(X, y)
ax[0].plot(range(1, len(a1n1.cost) + 1), np.log10(a1n1.cost), marker='o')
ax[0].set_xlabel('Эпохи')
ax[0].set_ylabel('log(Сумма квадратичных ошибок)')
ax[0].set_title('ADALINE. Теип обучения 0.01')
#Обучение при rate=0.0001
a1n2 = AdaptiveLinearNeuron(0.0001, 10).fit(X, y)
ax[1].plot(range(1, len(a1n2.cost) + 1), a1n2.cost, marker='o')
ax[1].set_xlabel('Эпохи')
ax[1].set_ylabel('Сумма квадратичных ошибок')
ax[1].set_title('ADALINE. Темп обучения 0.0001')
plt.show()
#Стандартизируем обучающую выборку
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
#Обучение на стандартизированной выборке при rate = 0.01
aln = AdaptiveLinearNeuron(0.01, 10)
aln.fit(X_std, y)
#строим график зависимости стоимости ошибок от эпох обучения
plt.plot(range(1, len(aln.cost) + 1), aln.cost, marker='o')
plt.xlabel('эпохи')
plt.ylabel('сумма квадратичных ошибок')
plt.show()

#Строим области принятия решений
plot_decision_regions(X_std, y, clf=aln)
plt.title('ADALINE (градиентный спуск)')
plt.xlabel('Длина чашелистика [стандартизированная]')
plt.ylabel('Длина лепестка[стандартизированная]')
plt.legend(loc='upper left')
plt.show()

i1 = [0.25, 1.1] #
R1 = aln.predict(i1)
print('R1 = ', R1)

if (R1 == 1):
    print('R1 = вид цветка Iris setosa')
else:
    print('R1 = вид цветка Iris vesticolor')
