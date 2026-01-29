# Python & Mathematics for Data Science and Machine Learning
# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# AI-powered by GPT-5.x

# ---- [cell 1] ----------------------------------------
import numpy as np

np.set_printoptions(precision=3, suppress=True)

# ---- [cell 2] ----------------------------------------
def softmax_rows(S):
    S = S - S.max(axis=1, keepdims=True)
    E = np.exp(S)
    return E / (E.sum(axis=1, keepdims=True) + 1e-12)

# ---- [cell 3] ----------------------------------------
def attention(Q, K, V, causal=False):
    d = Q.shape[1]
    S = (Q @ K.T) / np.sqrt(d)
    if causal:
        n = S.shape[0]
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        S = S.copy(); S[mask] = -1e9
    A = softmax_rows(S)
    O = A @ V
    return O, A

# ---- [cell 4] ----------------------------------------
rs = np.random.default_rng(22)
n, d, dv = 6, 4, 2
X = rs.normal(size=(n, d))
Wq = rs.normal(size=(d, d))
Wk = rs.normal(size=(d, d))
Wv = rs.normal(size=(d, dv))
Q, K, V = X @ Wq, X @ Wk, X @ Wv
O_nc, A_nc = attention(Q, K, V, causal=False)
O_c, A_c = attention(Q, K, V, causal=True)
print('rowsum(non-causal) →', np.round(A_nc.sum(1)[:3], 6))
print('future mass (causal) →', float(np.triu(A_c,1).sum()))
