import matplotlib.pyplot as plt
import scipy as sp

X = range(1,100,1)

X1 = sp.zeros([100])
Y = sp.zeros([100])


for x in X:
    X1[x] = x
    Y[x] = x*x


plt.plot(X1, Y)

plt.ylabel('some numbers')

plt.show()
