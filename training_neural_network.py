# чтобы сеть смогла обучиться выполнять заданные ей действия надо:
# 1. выбрать один пункт из набора данных (обучающей выборки)
# 2. подсчитать все частные производные потери по весу или смещению
# 3. выполнить обновления каждого веса и смещения (алгоритм оптимизации под названием "стохастический градиентный спуск"
# 4. возвратиться к первому пункту

import numpy as np

def sigmoid(x): # функция активации sigmoid - f(x) = 1/ (1 + e ^ (-x)
    return 1 / (1 + np. exp(-x))

def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx) # производная от sigmoid- f'(x) = f(x) * (1 - f(x))

#
def mse_loss(y_true, y_pred): # расчет среднеквадратичной ошибки
    return ((y_true - y_pred) ** 2).mean() # y_true и y_pred являются массивами с одинаковой длиной

class OurNeuralNetwork:
    def __init__(self):
        self.w1 = np.random.normal() #вес
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        self.b1 = np.random.normal() #смещения
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x): # формируем выходы (х является массивом с двумя элементами)
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1) #
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def train(self, data, all_y_trues):
        learn_rate = 0.1
        epochs = 1000 # количество циклов во всем наборе данных
        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1 # выполняем обратную связь (нам понадобятся эти значения в дальнейшем)
                h1 = sigmoid(sum_h1)
                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)
                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1
                d_L_d_ypred = -2 * (y_true - y_pred) # подсчет частных производных (d_L_d_w1 представляет "частично L/ частично w1"
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1) # нейрон o1
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)
                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1) # нейрон h1
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2) #нейрон h2
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1 #обновляем вес и смещения, нейрон h1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3 #нейрон h2
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5 #нейрон о1
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))

data = np.array([
    [-2, -1],  #Alice
    [25, 6],  #Bob
    [17, 4],  #Charlie
    [-15, 6] #Diana
    ])
all_y_trues = np.array([
    1,  #Alice
    0,  #Bob
    0,  #Charlie
    1,  #Diana
])

network = OurNeuralNetwork()
network.train(data, all_y_trues)