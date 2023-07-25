import numpy as np
from scipy.stats import multivariate_normal

n_samples = 1000
k_classes = 2
g1_precentage = 0.7
g2_precentage = 0.3

real_u1 = np.array([-1, -1])
real_sigma1 = np.array([[0.8, 0], [0, 0.8]])

real_u2 = np.array([1, 1])
real_sigma2 = np.array([[0.75, -0.2], [-0.2, 0.6]])

X1 = np.random.multivariate_normal(real_u1, real_sigma1, int(n_samples * g1_precentage))
X2 = np.random.multivariate_normal(real_u2, real_sigma2, int(n_samples * g2_precentage))
X = np.concatenate((X1, X2), axis=0)
np.random.shuffle(X)

num_tries = 1000
converge = np.zeros((num_tries))

means_init_method = 'furthest point' # options: random point(Default), random space, furthest point
sigma_init_method = 'std' # options: I(Defualt), std

for n in range(num_tries):
    u_means = np.zeros((k_classes, X.shape[1]))
    sigma = np.full_like(np.zeros((k_classes, X.shape[1], X.shape[1])), fill_value=np.eye(X.shape[1]))
    pi = np.ones((k_classes,)) / k_classes

    if (means_init_method == 'random space'):    
        X_space_max = np.max(X, axis=0)
        X_space_min = np.min(X, axis=0)
        u_means = (np.random.rand(k_classes, X.shape[1]) * (X_space_max - X_space_min)) + X_space_min
    elif (means_init_method == 'furthest point'):
        u_means[0] = X[np.random.randint(0, n_samples), :]
        for k in range(k_classes - 1):
            dist = np.zeros((n_samples, 1))
            for i in range(k + 1):
                dist += np.sum((X - u_means[i]) ** 2, axis=1).reshape((n_samples, 1))
            u_means[k + 1] = X[np.argmax(dist), :]
    else:
        u_means = X[np.random.randint(0, n_samples, (2)), :]

    if (sigma_init_method == 'std'):
        sigma = sigma * np.std(X, axis=0)
        
    init_means = u_means

    max_iter = 1000
    epsilon_0 = 1e-8
    epsilon_diff = 1e-3
    w = np.zeros((n_samples, k_classes))

    for k in range(max_iter):
        last_u = u_means
        last_sigma = sigma
        
        xproba = np.zeros((n_samples,))
        
        for j in range(k_classes):
            jguessian = multivariate_normal(u_means[j], sigma[j])
            jproba = jguessian.pdf(X)
            w[:, j] = jproba * pi[j] 
            xproba += w[:, j]
        w = w / (xproba.reshape((n_samples, 1)) + epsilon_0)
        
        pi = np.mean(w, axis=0)
        u_means = (np.transpose(w) @ X) / ((pi.reshape((k_classes, 1)) * n_samples) + epsilon_0)
        
        if (np.max(np.sum(np.abs(u_means - last_u), axis=1)) < epsilon_diff and np.max(np.sum(np.sum(np.abs(sigma - last_sigma)))) < epsilon_diff):
            converge[n] = k
            break
        
        for j in range(k_classes):
            sigma[j] = np.transpose(w[:, j].reshape((n_samples, 1)) * (X - u_means[j])) @ (X - u_means[j])
            sigma[j] = sigma[j] / ((n_samples * pi[j]) + epsilon_0)
            
            
print("for ", means_init_method, " + ", sigma_init_method)

print("\nmean convergance: ", np.mean(converge))
print("\n")