'''
Channels
'''

import torch
from dl_utils.d2l.cnn import corr2d

print('0--------------------------------')

def corr2d_multi_in(X, K):
    # Iterate over the 0th dimension (the channel dimension) of X and K, 
    # and then add the results together
    return sum(corr2d(x, k) for x, k in zip(X, K))

X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
print(corr2d_multi_in(X, K))
print('1--------------------------------')

def corr2d_multi_in_out(X, K):
    # Iterate over the 0th dimension of K, and each time, perform cross-correlation operation 
    # on the 0th dimension of X with the corresponding kernel. 
    # Then, stack the results along the 0th dimension
    return torch.stack([corr2d_multi_in(X, k) for k in K], dim=0)

K = torch.stack((K, K + 1, K + 2), dim=0)
print(K.shape)
print(corr2d_multi_in_out(X, K))

def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    # Perform matrix multiplication
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))

X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
print(float(torch.abs(Y1 - Y2).sum()) < 1e-6)
