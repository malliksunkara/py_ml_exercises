import numpy as np
import os
import matplotlib.pyplot as plt

def get_data(filename):
    data = np.genfromtxt(filename, delimiter=',')

    N_attributes = data.shape[1] - 1

    X = data[:, :N_attributes]
    Y = data[:, N_attributes]

    return X, Y

def normalize_data(Z):
    mu = np.mean(Z, 0)
    sigma = np.std(Z, 0)
    Zn = (Z - mu) / sigma

    return Zn

def add_bias(X):
    H = np.ones([X.shape[0], X.shape[1] + 1])
    H[:, 1:] = X

    return H

def hypothesis(t, X):
    return X.dot(t.transpose())

def cost(t, X, Y):
    return sum((X.dot(t.transpose()) - Y) ** 2) / len(X)

def dJdt(t,X,Y,j):
    return (X.dot(t.transpose()) - Y).transpose().dot(X[:,j]) * 2 / len(X)

def DJ(t,X,Y):
    return (X.dot(t.transpose()) - Y).transpose().dot(X) * 2 / len(X)

def gradient_descent_step(t, X, Y, alpha):
    return t - alpha * DJ(t,X,Y)

def gradient_descent(t, X, Y, alpha, steps):
    t_history = [t]
    cost_history = [cost(t,X,Y)]

    for i in range(0, steps, 1):
        t = gradient_descent_step(t, X, Y, alpha)
        t_history += [t]
        cost_history += [cost(t,X,Y)]

    return t, t_history, cost_history

def plot_history(vector):
    u = np.arange(len(vector))
    for i in range(0, vector.shape[1], 1):
        plt.plot(u, vector[:,i])
    plt.show()

def plot_c_history(vector):
    u = np.arange(len(vector))
    plt.plot(u, vector)
    plt.show()

def prevision(t, X):
    global mu, sigma
    X = (X - mu)/sigma
    Y = np.ones(len(X)+1)
    Y[1:] = X
    return hypothesis(t, Y)

os.chdir('C:/PyExercises/ml-ex1/ex1')
X, Y = get_data('ex1data1.txt')

N_var = X.shape[1]
mu = np.mean(X,0)
sigma = np.std(X,0)
Xn = add_bias(normalize_data(X))

Theta = np.random.rand(N_var + 1)
#Theta = np.array([1000000, 10, -100000])

alpha = 0.02
steps = 300

Theta, Theta_H, cost_H = gradient_descent(Theta, Xn, Y, alpha, steps)

Theta_H = np.array(Theta_H)
cost_H = np.array(cost_H)
print(Theta)

plot_history(Theta_H)
plot_c_history(cost_H)
