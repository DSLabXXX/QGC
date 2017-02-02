import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator


def mv(v):
    print(v)
    print(v.shape)
    return np.array([2*v[0], 3*v[1]])

A = LinearOperator((2, 2), matvec=mv)

print('A', A)
print('A.matvec', A.matvec(np.ones(2)))
print(np.ones(2))
# print('A*one', A * np.ones(2))

a = np.random.rand(2, 1)

def f(v):
    return np.dot(a, v)

A = LinearOperator(a.shape, f)

b = np.random.rand(2, 1)
# print(A.matvec(b))
# print(np.dot(a, b))