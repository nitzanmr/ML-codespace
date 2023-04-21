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
    # Matrix form calculations:
    # u = (np.transpose(data) @ (np.ones((data.shape[0], 1)) / data.shape[0]))
    # u = u.reshape(data.shape[1])
    # v = (np.transpose((u - data) ** 2) @ (np.ones((data.shape[0], 1)) / data.shape[0]))
    # v = v.reshape(data.shape[1])
    # using numpy:
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

def GD(x, y, theta0, max_iter=20, alpha=0.1, J_epsi=10**-8, theta_epsi=10**-8):
    last_theta = theta0
    last_cost = J(x, y, theta0)
    for k in range(max_iter):
        theta = last_theta - alpha * dJ(x, y, last_theta)
        new_cost = J(x, y, theta)
        if (np.abs(theta - last_theta) < theta_epsi or np.abs(new_cost - last_cost) < J_epsi):
            break
        last_cost = new_cost
        last_theta = theta
    return theta
        
def miniBatch(x, y, theta0, max_iter=20, alpha=0.1, J_epsi=10**-8, theta_epsi=10**-8, N=50):
    last_theta = theta0
    last_cost = J(x, y, theta0)
    for k in range(max_iter):
        theta = last_theta - alpha * dJ(x[:,N*k:N*(k+1)], y[N*k:N*(k+1)], last_theta)
        new_cost = J(x, y, theta)
        if (np.abs(theta - last_theta) < theta_epsi or np.abs(new_cost - last_cost) < J_epsi):
            break
        last_cost = new_cost
        last_theta = theta
    return theta

def Adam(x, y, theta0, max_iter=20, alpha=0.1, b=0.9, J_epsi=10**-8, theta_epsi=10**-8):
    epsi = 10**-8
    alphaj = np.ones((x.shape[1], 1))
    g = np.zeros((x.shape[1], 1))
    v = np.zeros((x.shape, 1))
    last_theta = theta0
    last_cost = np.mean(np.square(x @ theta0 - y))
    for k in range(max_iter):
        new_cost = np.mean(np.square(x @ last_theta - y))
        grad = (np.transpose(x)) @ ((x @ last_theta) - y)
        v = (b * v) + (1-b) * grad
        g = g + np.square(grad)
        alphaj = alpha / np.sqrt(g + epsi)
        theta = theta - alphaj * v
        if (np.abs(theta - last_theta) < theta_epsi or np.abs(new_cost - last_cost) < J_epsi):
            break
        last_cost = new_cost
        last_theta = theta
    return theta


data = np.loadtxt(open("cancer_data.csv", "rb"), delimiter=",")
nData, u, v = fixData(data)

print(nData.shape)

epsi = 10**-3
td, tu, tv = fixData(nData) #test that the average is indeed 0 and sigma is 1
for i in range(tu.size):
    print(i)
    print(tu[i])
    assert tu[i] <= epsi
    print(np.abs(tv[i] - 1))
    assert np.abs(tv[i] - 1) <= epsi
assert tu.all < epsi

print("normalization of data successfull")

x = np.delete(nData, -1, axis=1)
x = np.hstack((np.ones((x.shape[0],1)), x))
y = nData[:,-1]
y = y.reshape((y.size, 1))

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