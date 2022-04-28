import numpy as np

#функция активации: f(x) = 1 / (1 + e^(-x))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#создание класса Нейрон
class Neuron:
    def __init__(self, w):
        self.w = w


    def y(self, x): # сумматор
        s = np.dot(self.w, x) # суммируем входы
        return sigmoid(s) # функция активации

Xi = np.array([0, 0, 1, 1]) # задание значений входам
Wi = np.array([5, 4, 3, 1]) # веса входных сенсоров
n = Neuron(Wi) # создание объекта из класса нейрон
print('Y = ', n.y(Xi)) # обращение к нейрону