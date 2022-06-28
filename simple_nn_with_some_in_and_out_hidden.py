#without numpy
def w_sum(a, b):
    assert(len(a) == len(b))
    output = 0
    for i in range(len(a)):
        output += (a[i] * b[i])

    return output


def vect_mat_mul(vect, matrix):
    assert(len(vect) == len(matrix))
    output = [0, 0, 0]

    for i in range(len(vect)):
        output[i] = w_sum(vect, matrix[i])

    return output

def neural_network(input, weights):
    hid = vect_mat_mul(input, weights[0])
    pred = vect_mat_mul(hid, weights[1])
    return pred


toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]

input = [toes[0], wlrec[0], nfans[0]] #соответствует первой игре в сезоне

            #игр % побед #болельщиков
ih_weight =  [ [0.1, 0.2, -0.1],  #hid[0]
               [-0.1, 0.1, 0.9], #hid[1]
               [0.1, 0.4, 0.1] #hid[2]
             ]

          #hid[0] #hid[1] #hid[2]
hp_weight = [ [0.3, 1.1, -0.3], #травмы?
               [0.1, 0.2, 0.0], #победа?
               [0.0, 1.3, 0.1] #печаль?
             ]

weights = [ih_weight, hp_weight]

print('Result or nn without numpy:')
print(neural_network(input, weights))

#with numpy
import numpy as np

def neural_network_numpy(input, weights_numpy):
    hid = input.dot(weights_numpy[0])
    pred = hid.dot(weights_numpy[1])
    return pred


ih_weight_numpy = np.array([ [0.1, 0.2, -0.1],  #hid[0]
               [-0.1, 0.1, 0.9], #hid[1]
               [0.1, 0.4, 0.1] #hid[2]
             ]).T

hp_weight_numpy = np.array([
               [0.3, 1.1, -0.3], #травмы?
               [0.1, 0.2, 0.0], #победа?
               [0.0, 1.3, 0.1] #печаль?
             ]).T

weights_numpy = [ih_weight_numpy, hp_weight_numpy]

toes = np.array([8.5, 9.5, 9.9, 9.0])
wlrec = np.array([0.65, 0.8, 0.8, 0.9])
nfans = np.array([1.2, 1.3, 0.5, 1.0])

input_numpy = np.array([toes[0], wlrec[0], nfans[0]])

print('Result or nn with numpy:')
pred = neural_network_numpy(input_numpy, weights_numpy)
print(pred)