import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random as rnd
from scipy.optimize import fmin_bfgs


def show_samples(indices, n_columns=4):
    global X

    n_samples = len(indices)
    n_row =  int(n_samples / n_columns) + int(n_samples % n_columns != 0)
    pad = 1

    to_show = np.ones([20*n_columns + pad*(n_columns-1), 20*n_row + pad*(n_row-1)])

    for i in range(0, n_samples, 1):
        row = int(i/n_columns)
        col = i%n_columns
        posx = 20 * col + pad * col
        posy = 20 * row + pad * row
        to_show[posx:posx+20, posy:posy+20] = np.reshape(X[indices[i]], [20,20])

    plt.imshow(to_show.transpose())
    plt.show()

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def hypothesis(t, x):
    return sigmoid(x.dot(t.transpose()))

def cost(t, x, y, reg_factor):
    s1 = hypothesis(t,x)
    s2 = - y * np.log(s1) - (1-y) * np.log(1-s1)
    reg_term = reg_factor / (2*len(y)) * sum(t[1:] ** 2)
    to_return = sum(s2)/len(y) + reg_term
    return to_return

def DJ(t, x, y, reg_factor):
    s1 = hypothesis(t,x)
    s2 = s1 - y
    M = len(y)
    to_return = x.transpose().dot(s2) / M
    reg_term = np.zeros_like(to_return)
    reg_term[1:] = reg_factor / M * t[1:]
    to_return += reg_term
    return to_return

def convert_y_to_io(y, a):
    y2 = (y == a)
    to_return = np.zeros(len(y))
    for i in range(0, len(y), 1):
        if y2[i]: to_return[i] = 1
    return to_return

def train_one_vs_all(X, y, a):
    y = convert_y_to_io(y, a)
    theta = np.zeros(X.shape[1])
    regularizing_f = 1

    def local_cost(t):
        return cost(t, X, y, regularizing_f)

    def local_DJ(t):
        return DJ(t, X, y, regularizing_f)

    theta = fmin_bfgs(local_cost, theta, fprime=local_DJ)

    return theta

def check_on_samples(t, n_samples):
    global K, X, y

    correct_guess = 0
    indices = np.zeros(n_samples)
    for i in range(0, n_samples, 1):
        indices[i] = rnd.randint(0, 4999)
        ev = np.zeros(K)
        for a in range(0, K, 1):
            ev[a] = hypothesis(t[a], X[indices[i]])
            print('Sample ', i, 'is a "', a, '" with a chance of', ev[a])
        guess = ev.argmax()
        if guess == y[indices[i]]:
            correct_guess += 1
        print('I think it is a ', guess)

    show_samples(indices, n_columns=5)
    print('Correct guesses: ', correct_guess, 'out of ', n_samples, 'Percentage: ', correct_guess/n_samples * 100)


os.chdir('C:/PyExercises/ml-ex3/ex3')
mat = scipy.io.loadmat('ex3data1.mat')

X = mat['X']
y = mat['y']
K = 10

#indices = np.zeros(100)
#for i in range(0, len(indices), 1):
#    indices[i] = rnd.randint(0, 4999)
#show_samples(indices, n_columns=10)

THETA = np.zeros([K, X.shape[1]])

for a in range(0, K, 1):
    if a == 0: b=10
    else: b=a
    THETA[a] = train_one_vs_all(X, y, b)

check_on_samples(THETA, 100)