import scipy.io
import scipy as sp
import numpy as np
import os
import matplotlib.pyplot as plt


def get_data(filename):
    f_data = open(filename)
    raw_data = f_data.readlines()
    f_data.close()

    N = len(raw_data)
    X = dict()
    Y = dict()

    for line in range(0, N, 1):
        foundcomma = False
        X[line] = ''
        Y[line] = ''

        for c in raw_data[line]:
            if c == ',':
                foundcomma=True
            else:
                if not foundcomma:
                    X[line] += c
                else:
                    Y[line] += c

        X[line] = float(X[line])
        Y[line] = float(Y[line])

    return X, Y

def linear_function(theta, x):
    return theta[1]*x + theta[0]

def cost_function(theta, X, Y):
    N = len(X)
    H = dict()
    for i in X.keys():
        H[i] = (theta[1]*X[i] + theta[0] - Y[i])**2
    J = 1/N * sum(H.values())
    return J

def update_theta(theta, alpha, X, Y):
    N = len(X)
    H0 = dict()
    H1 = dict()
    for i in X.keys():
        H0[i] = (theta[1]*X[i] + theta[0] - Y[i])
        H1[i] = (theta[1]*X[i] + theta[0] - Y[i]) * X[i]

    dJdt0 = 2/N * sum(H0.values())
    dJdt1 = 2/N * sum(H1.values())

    ut0 = theta[0] - alpha * dJdt0
    ut1 = theta[1] - alpha * dJdt1

    return ut0, ut1

os.chdir('C:/PyExercises/ml-ex1/ex1')
X, Y = get_data('ex1data1.txt')

N = len(X)
Theta = dict()
Theta[0] = 0
Theta[1] = 1

Niter = 1000
alpha = 0.01

for i in range(0,Niter,1):
    Theta[0], Theta[1] = update_theta(Theta, alpha, X, Y)

print(Theta)

Xp = np.zeros(N)
Yp = np.ones(N)

for i in range(0,N,1):
    Xp[i] = X[i]
    Yp[i] = Y[i]

t = np.linspace(5,25,100)
r = np.zeros(len(t))
for q in range(0, len(t), 1):
    r[q] = Theta[1] * t[q] + Theta[0]

plt.plot(Xp,Yp, 'rx', t,r)
plt.show()

