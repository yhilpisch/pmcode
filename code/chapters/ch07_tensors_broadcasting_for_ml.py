# Python & Mathematics for Data Science and Machine Learning
# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# AI-powered by GPT-5.x

# ---- [cell 1] ----------------------------------------
# [magics stripped] %config InlineBackend.figure_format = 'retina'
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
rs = np.random.default_rng(0)

# ---- [cell 2] ----------------------------------------
a = np.arange(3.0).reshape(-1,1)
b = np.arange(4.0).reshape(1,-1)
grid = a + b
print(grid)

# ---- [cell 3] ----------------------------------------
B,M,K,N = 2,3,2,4
X = rs.normal(size=(B,M,K)).astype(np.float64)
Y = rs.normal(size=(B,K,N)).astype(np.float64)
Z = np.einsum('bmk,bkn->bmn', X, Y)
print('shapes:', X.shape, Y.shape, Z.shape)

# ---- [cell 4] ----------------------------------------
# Grid sum via broadcasting vs einsum
A = np.arange(3.0).reshape(-1,1); B = np.arange(4.0).reshape(1,-1)
grid = A + B
grid_ein = np.einsum('i,j->ij', A.ravel(), B.ravel())
ok1 = np.allclose(grid, grid_ein)
# Batched matmul vs loop
B_,M,K,N = 3,2,2,3
X = rs.normal(size=(B_,M,K)); Y = rs.normal(size=(B_,K,N))
Z = np.einsum('bmk,bkn->bmn', X, Y)
Z_loop = np.stack([X[b] @ Y[b] for b in range(B_)], axis=0)
ok2 = np.allclose(Z, Z_loop)
ok1, ok2
