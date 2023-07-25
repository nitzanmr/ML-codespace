import numpy as np
from numpy.linalg import det
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

np.random.seed(42)
n_samples = 1000
g1_precentage = 0.7
g2_precentage = 0.3

real_u1 = np.array([-1, -1])
real_sigma1 = np.array([[0.8, 0], [0, 0.8]])

real_u2 = np.array([1, 1])
real_sigma2 = np.array([[0.75, -0.2], [-0.2, 0.6]])

X1 = np.random.multivariate_normal(real_u1, real_sigma1, int(n_samples * g1_precentage))
X2 = np.random.multivariate_normal(real_u2, real_sigma2, int(n_samples * g2_precentage))

plt.scatter(X1[:, :-1], X1[:, -1], s=10)
plt.scatter(X2[:, :-1], X2[:, -1], s=10)
plt.show()

X = np.concatenate((X1, X2), axis=0)
plt.scatter(X[:, :-1], X[:, -1], s=10)
plt.show()

np.random.shuffle(X)

k_classes = 2
pi = np.ones((k_classes,)) / k_classes
sigma = np.zeros((k_classes, ), dtype=object)
u_means = np.zeros((k_classes, X.shape[1]))

for i in range(k_classes):
    sigma[i] = np.eye((X.shape[1]))
    u_means[i] = X[np.random.randint(0, n_samples), :]
# sigma = sigma * np.std(X, axis=0)

plt.scatter(u_means[:, 0], u_means[:, 1], s=20)
    
w = np.zeros((n_samples, k_classes))    

for k in range(100):
    xproba = np.zeros((n_samples,))
    
    for j in range(k_classes):
        jguessian = multivariate_normal(u_means[j], sigma[j])
        jproba = jguessian.pdf(X)
        w[:, j] = jproba * pi[j] 
        xproba += w[:, j]
    w = w / xproba.reshape((n_samples, 1))
    
    pi = np.mean(w, axis=0)
    u_means = (np.transpose(w) @ X) / (pi.reshape((k_classes, 1)) * n_samples)
    for j in range(k_classes):
        sigma[j] = np.transpose(w[:, j].reshape((n_samples, 1)) * (X - u_means[j])) @ (X - u_means[j])
        sigma[j] = sigma[j] / (n_samples * pi[j])
        
print("found means: \n", u_means)

X_pred = np.array((k_classes, 1), dtype=object)
found = np.zeros((k_classes,), dtype=bool)
guessians = np.empty((k_classes,), dtype=object)
proba = np.zeros((k_classes, n_samples))

for k in range(k_classes):
    guessians[k] = multivariate_normal(u_means[k], sigma[k])
    proba[k] = guessians[k].pdf(X)

max_index = np.argmax(proba, axis=0)

for i in range(n_samples):
    if not found[max_index[i]]:
        X_pred[max_index[i]] = X[i, :].reshape(1, X.shape[1])
        found[max_index[i]] = True
    else:
        X_pred[max_index[i]] = np.concatenate((X_pred[max_index[i]], X[i, :].reshape((1, X.shape[1]))), axis=0)

for k in range(k_classes):
    plt.scatter(X_pred[k][:, 0], X_pred[k][:, 1], s=10)
plt.scatter(u_means[:, :-1], u_means[:, -1], s=20)
plt.show()
