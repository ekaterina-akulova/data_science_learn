import numpy as np

# функция активации
def sigmoid(x):
    return 1/ (1 + np.exp(-x))

# описание класса нейрон
class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

#
class OurNeuralNetwork:
    def __init__(self):
        weights = np.array([0, 1]) # веса (одинаковы для всех нейронов)
        bias = 0 # смещение (одинаково для всех нейронов)
        self.h1 = Neuron(weights, bias) # формируем сеть из трех нейронов
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedforward(self, x):
        out_h1 = self.h1.feedforward(x) #формируем выход Y1 из нейрона h1
        out_h2 = self.h2.feedforward(x) #формируем выход Y2 из нейрона h2
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2])) #формируем выод из нейрона о1

        return out_o1


network = OurNeuralNetwork()  # создаем обьект СЕТЬ из класса OurNeuralNetwork
x = np.array([2, 3])  # формируем входные параметры для сети X1=2, X2=3
print("Y = ", network.feedforward(x)) # передаем входы в сеть и получаем результат