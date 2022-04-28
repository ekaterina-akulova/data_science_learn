import random
# искусственный нейрон (персептрон)
# def perseptron(Sensor):
#     b = 7 # порог функции активации
#     s = 0 # начальное значение суммы
#     for i in range (15): # цикл суммирования сигналов от сенсоров
#         s += int(Sensor[i]) * weights[i]
#
#     if s >= b:
#         return True # сумма превысила порог
#     else:
#         return False # сумма меньше порога
#
# # проверка работы искусственного нейрона (персептрона)
# num1 = list('001001001001001')
# num2 = list('111001111100111')
# weights = [1 for i in range(15)] # присваивание значений всем связям = 1
#
# print(num1) # смотрим, какие сигналы от сенсоров получил персептрон
# print(perseptron(num1)) # проверяем, что ответиил персепптрон
# print(num2) # смотрим, какие сигналы от сенсоров получил персептрон
# print(perseptron(num2)) # проверяем, что ответил персептрон

#обучающая выборка (идеальное изображение цифр от 0 до 9)
num0 = list('111101101101111')
num1 = list('001001001001001')
num2 = list('111001111100111')
num3 = list('111001111001111')
num4 = list('101101111001001')
num5 = list('111100111001111')
num6 = list('111100111101111')
num7 = list('111001001001001')
num8 = list('111101111101111')
num9 = list('111101111001111')

#спиок всех цифр от 0 до 9 в едином массиве
nums = [num0, num1, num2, num3, num4, num5, num6, num7, num8, num9]
tema = 5 # какой цифре обучаем
n_sensor = 15 # количество сенсоров

# инициализация весов для связей сенсоров с сумматором
weights = []
for i in range(15):
    weights.append(0)

# функция определяет, является ли полученное изображение числом 5
# возвращает да, если признано, что это 5, нет, если отвергнуто
def perceptron(Sensor):
    b = 7 # порог функции активации
    s = 0 # начальное значение суммы
    for i in range(n_sensor):
        s += int(Sensor[i]) * weights[i]
    if s >= b:
        return True # сумма превысила порог
    else:
        return False # сумма меньше порога

# Уменьшение значений весов
# Если сеть ошиблась и выдала да при входной цифре, отличной от пятерки
def decrease(number):
    for i in range(n_sensor):
        if int(number[i]) == 1: # Возбужденный ли вход
            weights[i] -= 1 # Уменьшаем связанный с входом вес на единицу

# Увеличение значений весов
# Если сеть ошиблась и выдала да при входной цифре, отличной от пятерки
def increase(number):
    for i in range(n_sensor):
        if int(number[i]) == 1: # Возбужденный ли вход
            weights[i] += 1 # Увеличиваем связанный с входом вес на единицу

#Тренировка сети
n = 1000 #
for i in range(n):
    j = random.randint(0, 9) # генерируем случайное число от 0 до 9
    r = perceptron(nums[j]) # результат обращения к сумматору (да - нет)
    if j != tema: # если генератор выдал случайное число j, не равное 5
        if r: # если сумматор сказал да (это пятерка), а j это не пятерка
            decrease(nums[j]) # наказываем сеть (уменьшаем значения весов)
    else: # если генератор выдал случайное число j, не равное 5
         if not r: # если сумматор сказал нет (т е Да, это пятерка) Не ошибся
            increase(nums[tema]) # поощряем сеть (увеличиваем значения весов)

print(weights) # Вывод значений весов
print("0 is 5?", perceptron(num0))
print("1 is 5?", perceptron(num1))
print("2 is 5?", perceptron(num2))
print("3 is 5?", perceptron(num3))
print("4 is 5?", perceptron(num4))
print("5 is 5?", perceptron(num5))
print("6 is 5?", perceptron(num6))
print("7 is 5?", perceptron(num7))
print("8 is 5?", perceptron(num8))
print("9 is 5?", perceptron(num9))

# различные варианты 5
num51 = list('111100111000111')
num52 = list('111100010001111')
num53 = list('111100011001111')
num54 = list('110100111001111')
num55 = list('110100111001011')
num56 = list('111100101001111')


print("5 is 51?", perceptron(num51))
print("5 is 52?", perceptron(num52))
print("5 is 53?", perceptron(num53))
print("5 is 54?", perceptron(num54))
print("5 is 55?", perceptron(num55))
print("5 is 56?", perceptron(num56))