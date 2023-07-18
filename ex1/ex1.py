import numpy as np
import matplotlib.pyplot as plt
import enum as Enum

#clean = Enum('clean', ['NOTHING','DELETE', 'FIX'])
#clean['NOTHING']

def NormalNarrow(num, u, s):
    if ((num - u) / s > 3):
        return (3 * s) + u
    elif (num < -3):
        return (-3 * s) + u
    return num

def fixData(data, clean=0):
    u = np.mean(data, axis=0)
    v = np.mean(((u - data) ** 2), axis=0)
    s = np.sqrt(v)
    fData = data
    if (clean != 0):
        if (clean == 1):
            vectorNarrow = np.vectorize(NormalNarrow)
            fData = vectorNarrow(fData, u, s)
        elif (clean == 2):
            fdata = fdata[np.max(np.abs(fData),axis=1) < (3 * s) + u]
        u = np.mean(data, axis=0)
        v = np.mean(((u - data) ** 2), axis=0)
    nData = (fData - u) / s

    return nData, u, v

def h(x, theta):
    xTemp = x.reshape((1, x.size))
    thetaTemp = theta.reshape((theta.size, 1))
    return xTemp @ thetaTemp

def J(x, y, theta):
    return (np.mean(np.square(x @ theta - y))) / 2

def dJ(x, y, theta):
    return np.transpose(x) @ ((x @ theta) - y)

def GD(x, y, theta0, max_iter=20, alpha=0.1, J_epsi=10**-3, theta_epsi=10**-3):
    last_theta = theta0
    last_cost = J(x, y, theta0)
    all_cost = []
    for k in range(max_iter):
        theta = last_theta - alpha * dJ(x, y, last_theta) / y.size
        new_cost = J(x, y, theta)
        all_cost.append(J(x,y,theta))
        # print(new_cost)
        if (False not in (np.abs(theta - last_theta) < theta_epsi) or np.abs(new_cost - last_cost) < J_epsi):
            if False not in (np.abs(theta - last_theta) < theta_epsi):
                print("theta ended")
            if (np.abs(new_cost - last_cost) < J_epsi):
                print("J ended")
            break
        last_cost = new_cost
        last_theta = theta
    return theta , k , all_cost
        
def getBatch(x, y, fromK, toK, max):
    fromK = fromK % max
    toK = toK % max
    if (fromK < toK):
        xbatch = x[fromK : toK,:]
        ybatch = y[fromK : toK]
    else:
        xbatch = np.vstack((x[fromK : max , :], x[0: toK]))
        ybatch = np.vstack((y[fromK : max , :], y[0: toK]))
    return xbatch, ybatch

def miniBatch(x, y, theta0, max_iter=20, alpha=0.1, J_epsi=10**-5, theta_epsi=10**-5, N=30):
    last_theta = theta0
    last_cost = J(x, y, theta0)
    all_cost = []
    for k in range(max_iter):
        xbatch, ybatch = getBatch(x, y, k*N, (k+1)*N, y.size)
        theta = last_theta - alpha * dJ(xbatch, ybatch, last_theta) / N
        new_cost = J(x, y, theta)
        all_cost.append(new_cost)
        print(new_cost)
        if (False not in (np.abs(theta - last_theta) < theta_epsi) or np.abs(new_cost - last_cost) < J_epsi):
            if False not in (np.abs(theta - last_theta) < theta_epsi):
                print("theta ended")
            if (np.abs(new_cost - last_cost) < J_epsi):
                print("J ended")
            break
        last_cost = new_cost
        last_theta = theta
    return theta, k,all_cost

def Adam(x, y, theta0, max_iter=20, alpha=0.4, b=0.9, J_epsi=10**-5, theta_epsi=10**-5):
    epsi = 10**-8
    alphaj = np.ones((x.shape[1], 1))
    g = np.zeros((x.shape[1], 1))
    v = np.zeros((x.shape[1], 1))
    last_theta = theta0
    last_cost = np.mean(np.square(x @ theta0 - y))
    all_cost = []
    for k in range(max_iter):
        grad = (np.transpose(x)) @ ((x @ last_theta) - y)
        v = (b * v) + (1-b) * dJ(x, y, last_theta)
        g = g + np.square(grad)
        alphaj = alpha / np.sqrt(g + epsi)
        theta = last_theta - alphaj * v
        new_cost = J(x, y, theta)
        all_cost.append(new_cost)
        if (False not in (np.abs(theta - last_theta) < theta_epsi) or np.abs(new_cost - last_cost) < J_epsi):
            if False not in (np.abs(theta - last_theta) < theta_epsi):
                print("theta ended")
            if (np.abs(new_cost - last_cost) < J_epsi):
                print("J ended")
            break
        last_cost = new_cost
        last_theta = theta
    return theta, k, all_cost


data = np.loadtxt(open("cancer_data.csv", "rb"), delimiter=",")
nData, u, v = fixData(data)

# print(nData.shape)

epsi = 10**-8
tu = np.mean(nData, axis=0)
tv = np.mean(((tu - nData) ** 2), axis=0)
assert True not in (tu > epsi)
assert True not in (np.abs(tv - 1) > epsi)

# print("normalization of data successfull")

x = np.delete(nData, -1, axis=1)
x = np.hstack((np.ones((x.shape[0],1)), x))
y = nData[:,-1]
y = y.reshape((y.size, 1))
theta0 = np.zeros((x.shape[1], 1))
# theta, k = Adam(x, y,theta0 = theta0, max_iter=100)
# print(J(x, y, theta))
# print(k)

# theta = np.zeros((xsize, 1))
# b = 0.9 # 0.99
# alpha = 0.1 # 0.13
# alphaj = np.ones((xsize, 1))
# g = np.zeros((xsize, 1))
# v = np.zeros((xsize, 1))
# max_iter = 10
# epsi = 10**-8
# m = y.size
# cost = np.zeros((max_iter, 1))

# # cost = np.mean(np.square(x@theta-y))
# # print("cost at start: ")
# # print(cost)

# for k in range(max_iter):
#     cost[k] = np.mean(np.square(x @ theta - y))
#     grad = (np.transpose(x)) @ ((x @ theta) - y)
#     v = (b * v) + (1-b) * grad
#     g = g + np.square(grad)
#     alphaj = alpha / np.sqrt(g + epsi)
#     theta = theta - alphaj * v
#     # print(theta)

# # cost = np.mean(np.square(x@theta-y))
# # print("\ncost at end: ")
# # print(cost)
# # print(theta)

# paintX = range(max_iter)
# print('cost final: \nfrom ')
# print(cost[0])
# print("to ")
# print( cost[cost.size -1])
# (cost[0],cost[cost.size - 1])
# # print()
# print("theata final: \n")
# print(theta.shape[0])
# theta = theta.reshape(theta.shape[0])
# print(theta.shape)

# print(sigma * theta)

# plt.plot(cost)
# plt.savefig('cost_in_hwew.png')