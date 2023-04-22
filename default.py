import ex1
import numpy as np
def __main__():
  matrix = np.ones((5,5))
  for i in range(5):
    for j in range(5):
      matrix[i][j] = i * 5 + j
  matrix,u,v = ex1.fixData(matrix)
  x = matrix[:, :-1]
  x = np.hstack((np.ones((x.shape[0],1)), x))
  y = matrix[:,-1]
  # print(y)
  # y = y.reshape(5,1)
  # print(y.shape)
  # print(x)
  # print('\n')
  theta = np.zeros(x.shape[1])
  
  # print(theta.reshape(5,1))
  # theta = theta.reshape(5,1)
  # print(theta.shape)
  theta = ex1.GD(x,y,theta,alpha=0.05)
  print(theta)
  print(y)


__main__()
