import numpy as np
import os
import matplotlib.pyplot as plt
import math
from scipy.optimize import fmin


def get_data(filename):
    data = np.genfromtxt(filename, delimiter=',')

    N_attributes = data.shape[1] - 1

    X = data[:, :N_attributes]
    Y = data[:, N_attributes]

    return X, Y

def add_bias(X):
    H = np.ones([X.shape[0], X.shape[1] + 1])
    H[:, 1:] = X

    return H

def detect_args(array, value):
    args = []
    for i in range(0, len(array), 1):
        if array[i] == value:
            args += [i]

    return np.array(args)

def plot_data(X, Y):
    X = X[:, 1:]
    positive_args = detect_args(Y, 1)
    negative_args = detect_args(Y, 0)
    X_positive = X[positive_args, :]
    X_negative = X[negative_args, :]

    plt.plot(X_positive[:,0], X_positive[:,1], 'ro', X_negative[:,0], X_negative[:,1], 'bx')
    plt.legend(('Ammessi', 'Non ammessi'))
    plt.show()

def plot_data_plus(X, Y, t):
    X = X[:, 1:]
    positive_args = detect_args(Y, 1)
    negative_args = detect_args(Y, 0)
    X_positive = X[positive_args, :]
    X_negative = X[negative_args, :]
    u = X[:, 0]
    v = - (t[1] * u + t[0]) / t[2]

    plt.plot(X_positive[:,0], X_positive[:,1], 'ro', X_negative[:,0], X_negative[:,1], 'bx',
             X[:,0], v, 'g'
             )
    plt.legend(('Ammessi', 'Non ammessi'))
    plt.show()


def sigmoid(z):
    return 1/(1 + math.exp(-z))

def sigmoid_v(vector):
    res = np.zeros(vector.shape)
    for i in range(0, len(vector), 1):
        res[i] = sigmoid(vector[i])
    return res

def hypothesis(t, x):
    return sigmoid(x.dot(t.transpose()))

def cost(t, x, y):
    lc = 0
    for j in range(0, len(x), 1):
        lc += (-1) * y[j] * np.log(hypothesis(t, x[j])) - (1 - y[j]) * np.log(1 - hypothesis(t, x[j]))
    lc = lc / len(x)
    return lc

def dJdt(t, x, y, j):
    res = 0
    for i in range(0,len(x), 1):
        res += (hypothesis(t, x[i]) - y[i]) * x[i,j]
    res = res/len(x)
    return res

def DJ(t, x, y):
    res = []
    for j in range(0, len(t), 1):
        res += [dJdt(t, x, y, j)]
    return np.array(res)

def gradient_descent_step(t, X, Y, alpha):
    return t - alpha * DJ(t,X,Y)

def gradient_descent(t, X, Y, alpha, steps):
    t_history = [t]
    cost_history = [cost(t,X,Y)]

    for i in range(0, steps, 1):
        t = gradient_descent_step(t, X, Y, alpha)
        t_history += [t]
        cost_history += [cost(t,X,Y)]

    return t, np.array(t_history), np.array(cost_history)

def plot_history(vector):
    u = np.arange(len(vector))
    for i in range(0, vector.shape[1], 1):
        plt.plot(u, vector[:,i])
    plt.legend(('0', '1'))
    plt.show()


os.chdir('C:/PyExercises/ml-ex2/ex2')
X, Y = get_data('ex2data1.txt')
X = add_bias(X)

Theta = np.zeros(X.shape[1])
#Theta[0] = -18

#alpha = 0.001
#steps = 100

#print(cost(Theta, X, Y))
#Theta, Theta_H, H_cost = gradient_descent(Theta, X, Y, alpha, steps)

def cost_fix(t):
    global X, Y
    return cost(t, X, Y)

t_opt = fmin(cost_fix, Theta)

print(t_opt)
print(hypothesis(t_opt, np.array([1, 45, 85])))

#print(H_cost)

#plot_history(Theta_H)
plot_data_plus(X,Y,t_opt)
