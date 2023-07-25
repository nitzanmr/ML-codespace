import numpy as np
from numpy.linalg import det
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# X = np.array([[1, 3], [5, -1], [1, 1], [-3, 4], [2, -2]])
# print("X: \n", X)

# u_means = X[np.random.randint(0, 5, (2)), :]

# print(u_means)

# sigma = np.zeros((2, 2, 2))
# sigma = np.full_like(np.zeros((2, 2, 2)) , fill_value=np.eye(2))

# sigma = np.full_like((2, 1), fill_value=np.eye(2), dtype=object)
# print(sigma)
# print(sigma[0])
# print(sigma[1])

# a = -2
# b = 3

# randArr = np.zeros((100, ))
# rand_field = np.zeros((100, ))

# randArr = np.random.rand(100, 20)
# rand_field = randArr * (b - a) + a

# print(randArr.shape)

# print("max rand: ", np.max(randArr))
# print("max space: ", np.max(rand_field))
# print("min rand: ", np.min(randArr))
# print("min space: ", np.min(rand_field))

# X = np.array([[1, 3], [5, -1], [1, 1], [-3, 4], [2, -2]])
# print("X: \n", X)

# u_means = np.zeros((2, 2))
# for k in range(2):
#     if (k == 0):
#         u_means[k] = X[np.random.randint(0, 5), :]
#     else:
#         dist = np.zeros((5, 1))
#         for i in range(k):
#             print("\nX - ", u_means[0], ": \n", X - u_means[i])
#             print("\n(X - ", u_means[0], ")^2: \n", (X - u_means[i]) ** 2)
#             print("\nentering: ", np.sum((X - u_means[i]) ** 2, axis=1).shape, "\nto dist: ", dist.shape)
#             # print(X - u_means[i])
#             dist += np.sum((X - u_means[i]) ** 2, axis=1).reshape((5, 1))
#         print(dist.shape)
#         u_means[k] = X[np.argmax(dist), :]
# print(u_means)


# np.random.seed(42)
# n_samples = 1000
# g1_precentage = 0.7
# g2_precentage = 0.3

# real_u1 = np.array([-1, -1])
# real_sigma1 = np.array([[0.8, 0], [0, 0.8]])

# real_u2 = np.array([1, 1])
# real_sigma2 = np.array([[0.75, -0.2], [-0.2, 0.6]])

# X1 = np.random.multivariate_normal(real_u1, real_sigma1, int(n_samples * g1_precentage))
# X2 = np.random.multivariate_normal(real_u2, real_sigma2, int(n_samples * g2_precentage))

# plt.scatter(X1[:, :-1], X1[:, -1], s=10)
# plt.scatter(X2[:, :-1], X2[:, -1], s=10)
# plt.show()
# plt.close()

# X = np.concatenate((X1, X2), axis=0)
# plt.scatter(X[:, :-1], X[:, -1], s=10)
# np.random.shuffle(X)

# k_classes = 2

# pi = np.ones((k_classes,)) / k_classes

# sigma = np.zeros((k_classes, X.shape[1], X.shape[1]))
# sigma[0] = np.eye(X.shape[1])
# sigma[1] = np.eye(X.shape[1])
# sigma *= np.std(X, axis=0)

# u_means = np.zeros((k_classes, X.shape[1]))
# u_means[0] = X[np.random.randint(0, n_samples), :]
# u_means[1] = X[np.random.randint(0, n_samples), :]

# epsilon = 1e-8

# w = np.zeros((n_samples, k_classes))

# for k in range(100):
#     xproba = np.zeros((n_samples,))

#     for j in range(k_classes):
#         jguessian = multivariate_normal(u_means[j], sigma[j])
#         jproba = jguessian.pdf(X)
#         w[:, j] = jproba * pi[j]
#         xproba += w[:, j]
#     w = w / (xproba.reshape((n_samples, 1)) + epsilon)

#     pi = np.mean(w, axis=0)
#     u_means = (np.transpose(w) @ X) / (pi.reshape((k_classes, 1)) * n_samples)
#     for j in range(k_classes):
#         sigma[j] = ((X - u_means[j]).T @ (w[:, j].reshape((-1, 1)) * (X - u_means[j]))) / (n_samples * pi[j])
#         print("sigma:", sigma[j])

# print("means: \n", u_means)
# print("sigma 1: \n", sigma[0])
# print("sigma 2: \n", sigma[1])

# plt.scatter(u_means[:, :-1], u_means[:, -1], s=20)
# plt.show()
# plt.close()

# X1_pred = np.array([[0, 0]])
# print(X1_pred.shape)
# X2_pred = np.array([[0, 0]])
# found = np.array([False, False, False])
# guessianG1 = multivariate_normal(u_means[0], sigma[0])
# guessianG2 = multivariate_normal(u_means[1], sigma[1])
# for i in range(n_samples):
#     if guessianG1.pdf(X[i, :]) > guessianG2.pdf(X[i, :]):
#         if not found[0]:
#             X1_pred[0] = X[i, :]
#             found[0] = True
#         else:
#             X1_pred = np.concatenate((X1_pred, X[i, :].reshape((1, 2))), axis=0)
#     elif guessianG1.pdf(X[i, :]) < guessianG2.pdf(X[i, :]):
#         if not found[1]:
#             X2_pred[0] = X[i, :]
#             found[1] = True
#         else:
#             X2_pred = np.concatenate((X2_pred, X[i, :].reshape((1, 2))), axis=0)

# plt.scatter(X1_pred[:, 0], X1_pred[:, 1], s=10)
# plt.scatter(X2_pred[:, 0], X2_pred[:, 1], s=10)
# plt.scatter(u_means[:, :-1], u_means[:, -1], s=20)
# plt.show()
