# при уменьшении смещения значение выходной функции уменьшается и уменьшается влияние персипторна на принятие решения и наоборот

import numpy as np

# функция активации
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# создание класса нейрон
class Neuron:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def y(self, x):
        s = np.dot(self.w, x) + self.b
        return sigmoid(s)

Xi = np.array([2, 3]) #задание значений входам
Wi = np.array([0, 1]) #веса входных сенсоров
bias = 4 #смещение
n = Neuron(Wi, bias) #создание обьекта из класса нейрон
print('Y=', n.y(Xi))