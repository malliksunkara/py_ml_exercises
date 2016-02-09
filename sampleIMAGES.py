import scipy.io
import scipy as sp
from os import *
import random

chdir('C:/Users/Pietro T/PycharmProjects/machine_l/sparseae_exercise/starter')

mat = scipy.io.loadmat('IMAGES.mat')

IMAGES = mat.get('IMAGES')

x_dim = IMAGES.shape[0]
y_dim = IMAGES.shape[1]
n_dim = IMAGES.shape[2]

patchsize = 8
numpatches = 10000

sample = sp.zeros((patchsize, patchsize, numpatches))

for j in range(0,10,1):
    for n in range(0,1000,1):
        x = random.randint(0, x_dim - patchsize)
        y = random.randint(0, y_dim - patchsize)
        sample[:,:, (1000*j + n)] = IMAGES[x:(x+patchsize), y:(y+patchsize), j]


