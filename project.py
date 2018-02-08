import numpy as np
import matplotlib.pyplot as plt
from math import exp

# dataset text files
train_inp = np.genfromtxt('trainset.csv', delimiter = ",")
train_actual = np.genfromtxt('trainset_actual.csv', delimiter = ",")
test_inp = np.genfromtxt('testset.csv', delimiter = ",")
test_actual = np.genfromtxt('testset_actual.csv', delimiter = ",")
train_1 = np.genfromtxt('atrain.csv', delimiter = ",")
train_2 = np.genfromtxt('btrain.csv', delimiter = ",")
train_3 = np.genfromtxt('ctrain.csv', delimiter = ",")
test_1 = np.genfromtxt('test1.csv', delimiter = ",")
test_2 = np.genfromtxt('test2.csv', delimiter = ",")
test_3 = np.genfromtxt('test3.csv', delimiter = ",")

round = 500 # epoch
exp1_topology = np.array([5,4,3])
exp2_topology = np.array([5,3,5])
exp2_2_topology = np.array([3,2,3]) # topology in classification neural network
exp1_activation = np.array([0,0]) # relu for 1, sigmoid for 0
exp2_activation = np.array([0,0])
momentum = 1 # high momentum 0.9
learning_rate = 1 # low learning 0.01

activated_values = []
weight = []
new_w = []
class_epoch_error = []
class_each_error = []
exp1_guess = []
train_encoder = []
test_encoder = []
exp2_guess = []
exp1_train_error = []
exp2_train_error = []

def weightactivation(topology):
    global weight
    weight = []
    for i in range(0, (topology.size - 1)):
        x = np.random.rand(topology[i], topology[i + 1])
        weight.append(x)
    weight = np.array(weight)
        
def relu(a):
    for index, num in enumerate(a):
        if num > 0:
           a[index] = num
        else:
            a[index] = 0

def sigmoid(a):
    for index, num in enumerate(a):
        a[index] = (1 / (1 + exp(-num)))


def feedforward(inp, weight, topology, activation):
    global activated_values
    activated_values = []
    g = inp.copy() # values are copied and a new object is made
    for i in range(0, (topology.size - 1)): # last element not included, topology.size - 1
        g = np.dot(g, weight[i])
        if activation[i] == 0: #supposed last element
            sigmoid(g)
        else:
            relu(g)
        #print(f"Activated at {i}: {g}")
        activated_values.append(g)
    activated_values = np.array(activated_values)
    return g

def error(actual, guess):
    #e = ((guess - actual)**2)*0.5
    e = guess - actual
    return e

def gradients(a, b):
    gradient = a*b
    return gradient

def derivofy(answer, index, activation):
    y = answer.copy()
    if activation[index] == 0:
        for i, num in enumerate(y):
            y[i] = num * (1- num)
    else:
        for i, num in enumerate(y):
            if num > 0:
                y[i] = 1
            else:
                y[i] = 0
    return y

def backprog(gradient, weight, r, topology, inp, activation):
    global new_w
    tempnew_w = []
    activated_values_counter = activation.size - 2
    weight_counter = activation.size - 1
    temp = np.array([activated_values[activated_values_counter]]) # beside G^T
    w_a = np.dot(gradient.T, temp)
    #print(f"Temp: {temp}")
    #print(f"W_a: {w_a}")
    l = weight.copy()
    l = l[weight_counter]* momentum #weights
    tempnew_w.append(l - w_a.T)
    #print(f"First new_w: {tempnew_w}")
    gh = gradient
    for i in range(0, activation.size - 1):
        #print(f"Z' Round: {i + 1}")
        z_prime = derivofy(activated_values[activated_values_counter], activated_values_counter, activation)
        #print(f"Z': {z_prime}")
        temp_w = np.dot(gh, l.T)
        #print(temp_w)
        gh = z_prime * temp_w
        #print(f"Gh: {gh}")
        activated_values_counter = activated_values_counter - 1
        if activated_values_counter < 0:
            temp_delta = np.array([inp[r].copy()])
            #print(f"Temp delta: {temp_delta}")
        else:
            temp_delta = np.array([activated_values[activated_values_counter]])
            #print(f"Temp delta: {temp_delta}")
        delta_w = np.dot(temp_delta.T, gh)
        delta_w = learning_rate * delta_w
        #print(f"Delta W: {delta_w}")
        weight_counter = weight_counter - 1
        l = weight.copy()
        l = l[weight_counter] * momentum
        tempnew_w.append(l - delta_w)
        #print(tempnew_w)
    new_w = tempnew_w
    return new_w

def ctraining_neuralnetwork(inp, actual, topology, class_epoch_error, activation):
    global new_w, weight
    weightactivation(topology)
    for i in range(0, round):
        #print(f"Epoch: {i + 1}")
        class_each_error = []
        for r in range(0, inp.shape[0]):
            #print(f"Datapoint: {inp[r]}")
            if (r == 0) and (i == 0):
                guess_ = feedforward(inp[r], weight, topology, activation)
            elif (i == round - 1):
                guess_ = feedforward(inp[r], new_w, topology, activation)

            else:
                guess_ = feedforward(inp[r], new_w, topology, activation)
                #print(new_w)
            #print(f"Guess: {guess_}")
            deriv_ = derivofy(guess_, activation.size - 1, activation)
            #print(f"Derivative of Y: {deriv_}")
            #print(f"Actual: {actual[r]}")
            error_ = error(actual[r], guess_)
            #print(f"Error: {error_}")
            #print(f"Sum of Error: {np.sum(error_)}")
            sum_error = np.sum(error_)
            class_each_error.append(sum_error)
            gradient_ = np.array([gradients(error_, deriv_)])
            #print(f"Gradient: {gradient_}")
            if (r == 0) and (i == 0):
                backprog(gradient_, weight, r, topology, inp, activation)
            else:
                backprog(gradient_, new_w, r, topology, inp, activation)
            new_w = np.flipud(new_w)
            #print(f"New Weights: {new_w}")
        #Last Feed Forward
            guess_ = feedforward(inp[r], new_w, topology, activation)
            #print(f"Guess: {guess_}")
            deriv_ = derivofy(guess_, activation.size - 1, activation)
            #print(f"Derivative of Y: {deriv_}")
            #print(f"Actual: {actual[r]}")
            error_ = error(actual[r], guess_)
            #print(f"Error: {error_}")
            #print(f"Sum of Error: {sum_error}")
            gradient_ = np.array([gradients(error_, deriv_)])
            #print(f"Gradient: {gradient_}")
        class_epoch_error.append(np.average(class_each_error))
        #print(class_each_error)

def classification_network(guess, array):
    for r in range(0, guess.shape[0]):
        rel_max = np.amax(guess[r])
        if guess[r][0] == rel_max:
           array.append(1)
        elif guess[r][1] == rel_max:
            array.append(2)
        elif guess[r][2] == rel_max:
            array.append(3)

def test_neuralnetwork(inp, weight, class_guess, topology, activation):
    guess_array = []
    for i in range(0, inp.shape[0]):
        guess_ = feedforward(inp[i], weight, topology, activation)
        guess_array.append(guess_)
    guess_array = np.array(guess_array)
    classification_network(guess_array, class_guess)

def autoencoder(inp, topology, encoder, activation):
    global new_w, weight
    weightactivation(topology)
    for i in range(0, round):
        #print(f"Epoch: {i + 1}")
        class_each_error = []
        for r in range(0, inp.shape[0]):
            #print(f"Datapoint: {inp[r]}")
            if (r == 0) and (i == 0):
                guess_ = feedforward(inp[r], weight, topology, activation)
            elif (i == round - 1):
                index = (int(topology.size / 2)) - 1
                encoder.append(activated_values[index])
                #print(activated_values[index])
            else:
                guess_ = feedforward(inp[r], new_w, topology, activation)
                #print(new_w)
            #print(f"Guess: {guess_}")
            deriv_ = derivofy(guess_, activation.size - 1, activation)
            #print(f"Derivative of Y: {deriv_}")
            #print(f"Actual: {actual[r]}")
            error_ = error(inp[r], guess_)
            #print(f"Error: {error_}")
            #print(f"Sum of Error: {np.sum(error_)}")
            sum_error = np.sum(error_)
            class_each_error.append(sum_error)
            gradient_ = np.array([gradients(error_, deriv_)])
            #print(f"Gradient: {gradient_}")
            if (r == 0) and (i == 0):
                backprog(gradient_, weight, r, topology, inp, activation)
            else:
                backprog(gradient_, new_w, r, topology, inp, activation)
            new_w = np.flipud(new_w)
            #print(f"New Weights: {new_w}")
        #Last Feed Forward
            guess_ = feedforward(inp[r], new_w, topology, activation)
            #print(f"Guess: {guess_}")
            deriv_ = derivofy(guess_, activation.size - 1, activation)
            #print(f"Derivative of Y: {deriv_}")
            #print(f"Actual: {actual[r]}")
            error_ = error(inp[r], guess_)
            #print(f"Error: {error_}")
            #print(f"Sum of Error: {sum_error}")
            gradient_ = np.array([gradients(error_, deriv_)])
            #print(f"Gradient: {gradient_}")
        class_epoch_error.append(np.average(class_each_error))

def accuracy(guess, actual):
    correct = 0
    for i in range(0, len(guess)):
        if guess[i] == actual[i]:
            correct = correct + 1
    accuracy = (correct / len(actual)) * 100
    return accuracy

#Experiment 1
ctraining_neuralnetwork(train_inp, train_actual, exp1_topology, exp1_train_error, exp1_activation)
exp1_weight = new_w
print(f"Experiment 1 Errors: {exp1_train_error}")
plt.plot(exp1_train_error, 'ro')
test_neuralnetwork(test_inp, exp1_weight, exp1_guess, exp1_topology, exp1_activation)

#Experiment 2-- Autoencoders
autoencoder(train_1, exp2_topology, train_encoder, exp2_activation)
autoencoder(train_2, exp2_topology, train_encoder, exp2_activation)
autoencoder(train_3, exp2_topology, train_encoder, exp2_activation)
train_encoder = np.array(train_encoder)
ctraining_neuralnetwork(train_encoder, train_actual, exp2_2_topology, exp2_train_error, exp1_activation)
print()
print(f" Experiment 2 Errors: {exp2_train_error}")
plt.plot(exp2_train_error, 'bo')
exp2_weight = new_w
autoencoder(test_1, exp2_topology, test_encoder, exp2_activation)
autoencoder(test_2, exp2_topology, test_encoder, exp2_activation)
autoencoder(test_3, exp2_topology, test_encoder, exp2_activation)
test_encoder = np.array(test_encoder)
test_neuralnetwork(test_encoder, exp2_weight, exp2_guess, exp2_2_topology, exp1_activation)
print(f"Experiment 1: {exp1_guess}")
print(f"Experiment 2: {exp2_guess}")
print(f"Actual: {np.array(test_actual)}")
exp1_accuracy = accuracy(exp1_guess, test_actual)
print(f"Experiment 1 Accuracy: {exp1_accuracy}%")
exp2_accuracy = accuracy(exp2_guess, test_actual)
print(f"Experiment 2 Accuracy: {exp2_accuracy}%")

# Graphing of Errors
plt.legend(['Experiment 1', 'Experiment 2'], loc = 'lower right')
plt.show()