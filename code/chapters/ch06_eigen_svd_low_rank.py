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
A = np.array([[3., 1.],[1., 2.]])
w, Q = np.linalg.eigh(A)
print('eigs=', np.round(w, 6))
print('orthonormal columns?', np.allclose(Q.T @ Q, np.eye(2)))

# ---- [cell 3] ----------------------------------------
M = np.array([[2., 0., 0.],[0., 1., 0.]])
U, S, Vt = np.linalg.svd(M, full_matrices=False)
k=1; Mk = U[:,:k] @ np.diag(S[:k]) @ Vt[:k,:]
print('S=', np.round(S, 6), ' error_F=', np.linalg.norm(M-Mk, 'fro'))

# ---- [cell 4] ----------------------------------------
errs = []
for k in [0,1]:
    Uk, Sk, Vk = U[:,:k], np.diag(S[:k]), Vt[:k,:]
    Mk = Uk @ Sk @ Vk
    lhs = np.linalg.norm(M - Mk, ord='fro')**2
    rhs = np.sum(S[k:]**2)
    errs.append(np.allclose(lhs, rhs))
errs
