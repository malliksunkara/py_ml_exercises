import numpy as np
import os
import matplotlib.pyplot as plt
import math
from scipy.optimize import fmin
from scipy.optimize import fmin_bfgs


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

def plot_contour(t):
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)

    z = np.zeros([len(u), len(v)])

    for i in range(0, len(u), 1):
        for j in range(0, len(v), 1):
            z[i,j] = create_feature_e(np.array([u[i], v[j]])).dot(t.transpose())

    plt.contour(u, v, z, 1)
    plt.show()

def plot_contour_plus(X, Y, t):
    X = X[:, 1:]
    positive_args = detect_args(Y, 1)
    negative_args = detect_args(Y, 0)
    X_positive = X[positive_args, :]
    X_negative = X[negative_args, :]

    #plt.legend(('Ammessi', 'Non ammessi'))

    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)

    z = np.zeros([t.shape[0], len(u), len(v)])

    for m in range(0, t.shape[0], 1):
        for i in range(0, len(u), 1):
            for j in range(0, len(v), 1):
                z[m,j,i] = create_feature_e(np.array([u[i], v[j]])).dot(t[m, :].transpose())

    plt.plot(X_positive[:,0], X_positive[:,1], 'yo', X_negative[:,0], X_negative[:,1], 'bx')
    for m in range(0, t.shape[0], 1):
        plt.contour(u, v, z[m], 1)
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

def cost(t, x, y, lamb):
    lc = 0
    for j in range(0, len(x), 1):
        lc += (-1) * y[j] * np.log(hypothesis(t, x[j])) - (1 - y[j]) * np.log(1 - hypothesis(t, x[j]))
    lc = lc / len(x)
    tred = t[1:]
    lc += lamb / (2*len(x)) * sum(tred**2)
    return lc

def dJdt(t, x, y, lamb, j):
    res = 0
    j_isnt_zero = int(j != 0)
    for i in range(0,len(x), 1):
        res += (hypothesis(t, x[i]) - y[i]) * x[i,j]
    res = res/len(x) + j_isnt_zero * lamb / len(x) * t[j]
    return res

def DJ(t, x, y, lamb):
    res = []
    for j in range(0, len(t), 1):
        res += [dJdt(t, x, y, lamb, j)]
    return np.array(res)

def cost_fix(t):
    global Xfeat, Y, lamba
    return cost(t, Xfeat, Y, lamba)

def DJ_fix(t):
    global Xfeat, Y, lamba

    res = np.zeros_like(t)
    for j in range(0, len(t), 1):
        res[j] = dJdt(t, Xfeat, Y, lamba, j)
    return res

def plot_history(vector):
    u = np.arange(len(vector))
    for i in range(0, vector.shape[1], 1):
        plt.plot(u, vector[:,i])
    plt.legend(('0', '1'))
    plt.show()

def create_feature(x, N=6):
    NN = (N+1) * (N+2) / 2
    xFeat = np.zeros([len(x), NN])
    xFeat[:, 0:3] = x
    index = 3
    for degree in range(2, N+1, 1):
        for k in range(0, degree+1, 1):
            xFeat[:, index] = x[:, 1] ** (degree - k) * x[:, 2] ** k
            index += 1
    return xFeat

def create_feature_e(x, N=6):
    NN = (N+1) * (N+2) / 2
    xFeat = np.zeros(NN)
    xFeat[0] = 1
    xFeat[1:3] =  x
    index = 3
    for degree in range(2, N+1, 1):
        for k in range(0, degree+1, 1):
            xFeat[index] = x[0] ** (degree - k) * x[1] ** k
            index += 1
    return xFeat

def many_lambdas(first, last, gap):
    global lamba, Xfeat, Y
    vect = np.linspace(first, last, gap)
    opt_thetas = np.zeros([gap, 28])
    for v in range(0,gap,1):
        Theta0 = np.zeros(Xfeat.shape[1])
        lamba = vect[v]
        opt_thetas[v] = fmin_bfgs(cost_fix, Theta0, fprime=DJ_fix)

    return opt_thetas


os.chdir('C:/PyExercises/ml-ex2/ex2')
X, Y = get_data('ex2data2.txt')
X = add_bias(X)

Xfeat = create_feature(X)
Theta = np.zeros(Xfeat.shape[1])
lamba = 0

TT = many_lambdas(10, 200, 5)

plot_contour_plus(X, Y, TT)