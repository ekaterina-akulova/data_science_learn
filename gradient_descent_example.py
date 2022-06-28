knob_weight = 0.5
input = 0.5
goal_pred = 0.8

#hot cold
#step_amount = 0.001
# for iteration in range(1101):
#     prediction = input * knob_weight
#     error = (prediction - goal_pred) ** 2
#
#    # print("Error:" + str(error) + " Prediction:" + str(prediction))
#     up_prediction = input * (knob_weight + step_amount) #try to +
#     up_error = (goal_pred - up_prediction) ** 2
#
#     down_prediction = input * (knob_weight - step_amount)
#     down_error = (goal_pred - down_prediction) ** 2
#
#     if (down_error < up_error):
#         knob_weight = knob_weight - step_amount
#     if (down_error > up_error):
#         knob_weight = knob_weight + step_amount


pred = input * knob_weight
error = (pred - goal_pred) ** 2 #делает чистую ошибку положительной, умножая ее на саму себя, так как отрицательные ошибки не имеют смысла.
#print(error) #чистая ошибка

for iteration in range(20):
    pred = input * knob_weight
    error = (pred - goal_pred) ** 2
    direction_and_amount = (pred - goal_pred) * input #чистая ошибка,  масштабирование, обращение знака и остановка
    knob_weight = knob_weight - direction_and_amount
    # print("Error:" + str(error) + " Prediction:" + str(pred))


#another example with alpha

weight, goal_pred, input = (0.0, 0.8, 1.1)
alpha = 0.01 #упавляет скоростью обучения сети. слишком быстрое обучение влечет слишком агрессивную корректировку весовых коэффициетов и приводит к большим промахам

for iteration in range(4):
    print("-----\nWeight:" + str(weight))
    pred = input * weight
    error = (pred - goal_pred) ** 2
    delta = pred - goal_pred #difference
    weight_delta = input * delta #определяет величину изменения веса, обусловленную промахом сети (производная). масштабирование разности на выходее взвешиваемым входом
    weight -= weight_delta * delta
    print("Error:" + str(error) + " Prediction:" + str(pred))
    print("Delta:" + str(delta) + " Weight Delta:" + str(weight_delta))


weight = 0.5
goal_pred = 0.8
input = 2
alpha = 0.1
for iteration in range(20):
   pred = input * weight
   error = (pred - goal_pred) ** 2
   derivative = input * (pred - goal_pred)
   weight = weight - (alpha * derivative)


#обучение мeтодом градиентного спуска с несколькими входами

def w_sum(a, b):
    assert (len(a) == len(b))
    output = 0
    for i in range(len(a)):
        output += (a[i] * b[i])
    return output


def neural_network(input, weights):
    pred = w_sum(input, weights)
    return  pred


def ele_mul(number, vector):
    output = [0, 0, 0]
    assert(len(output) == len(vector))
    for i in range(len(vector)):
        output[i] = number * vector[i]
    return output

weights = [0.1, 0.2, -.1]
toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]

win_or_lose_binary = [1, 1, 0, 1]
true = win_or_lose_binary[0]
input = [toes[0], wlrec[0], nfans[0]]
pred = neural_network(input, weights)
error = (pred - true) ** 2
delta = pred - true
weight_deltas = ele_mul(delta, input)
alpha = 0.01 #перед корректировкой веса приращение weight deltg умножается на небольшое число alpha, что позволяет управлять скоростью обучения сети

for i in range(len(weights)):
    weights[i] -= alpha * weight_deltas[i]
print("Weights:" + str(weights))
print("Weights Deltas:" + str(weight_deltas))


#рассмотрим несколько шагов обучения
def neural_network(input, weights):
    out = 0
    for i in range(len(input)):
        out += (input[i] * weights[i])
    return  out

weights = [0.1, 0.2, -.1]
input = [toes[0], wlrec[0], nfans[0]]
for iter in range(3):
    pred = neural_network(input, weights)
    error = (pred - true) ** 2
    delta = pred - true
    weight_deltas = ele_mul(delta,input)
    print(
    )
    print("Iteration:" + str(iter + 1))
    print("Pred:" + str(pred))
    print("Error:" + str(error))
    print("Delta:" + str(delta))
    print("Weights:" + str(weights))
    print("Weight_Deltas:")
    print(str(weight_deltas))
    print(
    )

    for i in range(len(weights)):
        weights[i] -= alpha * weight_deltas[i]