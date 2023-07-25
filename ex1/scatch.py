import numpy as np
import csv
import matplotlib.pyplot as plt

def J(x, y, theta):
    return (np.mean(np.square(x @ theta - y))) / 2

def dJ(x, y, theta):
    return np.transpose(x) @ ((x @ theta) - y)

# def getBatch(x, y, fromK, toK, max):
#     fromK = fromK % max
#     toK = toK % max
#     if (fromK < toK):
#         xbatch = x[fromK : toK,:]
#         ybatch = y[fromK : toK]
#     else:
#         xbatch = np.vstack((x[fromK : max , :], x[0: toK]))
#         ybatch = np.vstack((y[fromK : max , :], y[0: toK]))
#     return xbatch, ybatch

# def miniBatch(x, y, theta0, max_iter=20, alpha=0.1, J_epsi=10**-8, theta_epsi=10**-8, N=30):
#     last_theta = theta0
#     last_cost = J(x, y, theta0)
#     for k in range(max_iter):
#         xbatch, ybatch = getBatch(x, y, k*N, (k+1)*N, y.size)
#         theta = last_theta - alpha * dJ(xbatch, ybatch, last_theta) / N
#         new_cost = J(x, y, theta)
#         if (False not in (np.abs(theta - last_theta) < theta_epsi) or np.abs(new_cost - last_cost) < J_epsi):
#             if False not in (np.abs(theta - last_theta) < theta_epsi):
#                 print("theta ended")
#             if (np.abs(new_cost - last_cost) < J_epsi):
#                 print("J ended")
#             print(x[(N*k) % x.shape[0] : N*(k+1) % x.shape[0],:])
#             print(y[(N*k) % x.shape[0] : N*(k+1) % x.shape[0]])
#             break
#         last_cost = new_cost
#         last_theta = theta
#     return theta, k

def Adam(x, y, theta0, max_iter=20, alpha=0.1, b=0.9, J_epsi=10**-8, theta_epsi=10**-8):
    epsi = 10**-4
    g = np.zeros((x.shape[1], 1))
    last_theta = theta0
    v = np.zeros((x.shape[1], 1))
    v = dJ(x, y, last_theta)
    last_cost = np.mean(np.square(x @ theta0 - y))
    for k in range(max_iter):
        grad = dJ(x, y, last_theta)
        v = (b * v) + (1-b) * grad
        alphas = alpha / np.sqrt(g + epsi)
        print("\n")
        theta = last_theta - alphas * v #* y.size
        print(grad)
        print(alphas)
        # print(v)
        print(theta) 
        new_cost = J(x, y, theta)
        if (False not in (np.abs(theta - last_theta) < theta_epsi) or np.abs(new_cost - last_cost) < J_epsi):
            if False not in (np.abs(theta - last_theta) < theta_epsi):
                print("theta ended")
            if (np.abs(new_cost - last_cost) < J_epsi):
                print("J ended")
            break
        # print(g)
        g = g + grad ** 2
        last_cost = new_cost
        last_theta = theta
    return theta, k

real_theta = np.array([3,5])
m = 210
x=np.ones((m,2))
x[:,1] = np.linspace(0,3,m)
noise = np.random.randn(m)
y = x@real_theta+noise
y = y.reshape((m,1))

# g = np.ones((10))
# for i in range(10):
#     g[i] = g[i] * i + 1
    
# print(g)

# alphas = 1 / g
# print(alphas)

theta, k = Adam(x, y, theta0=np.zeros((2,1)), max_iter=10, alpha=0.1)
print(k)
print("\n")
print(theta)