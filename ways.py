import numpy as np
import os
import timeit

os.chdir('C:/PyExercises/ml-ex1/ex1')
data = np.genfromtxt('ex1data2.txt', delimiter=',')

asd = np.array([[1,2,3], [4,5,6], [1,1,1]])
print(asd.ndim)
print(asd.sum())

def mean(X):
    return sum(X)/len(X)

def devi(X):
    mu = mean(X)
    H = (X - mu) ** 2
    Q = sum(H) / len(H)

    return np.sqrt(Q)


