import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-8, 8, 0.1) #интервал  и шаг изменения сигнала от сенсора Х

b1 = -2 # значения смещения
b2 = 0
b3 = 2

l1 = 'b = -2'
l2 = 'b = 0'
l3 = 'b = 2'
for b, l in [(b1, l1), (b2, l2), (b3, l3)]: # организация цикла для трех значений смещения
    f = (1 / (1 + np.exp((-x+b) * 1))) # функция сигмоиды
    plt.plot(x, f, label=l)
plt.xlabel('x') # подпись оси х
plt.ylabel('Y = f(x)') # подпись оси у
plt.legend(loc=4) # место легенды на графике
plt.show()