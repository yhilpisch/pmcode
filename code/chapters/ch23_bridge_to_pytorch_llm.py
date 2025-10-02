# Python & Mathematics for Data Science and Machine Learning
# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# AI-powered by GPT-5

"""Generated from notebook: /Users/yves/Library/CloudStorage/Dropbox/Program/books/4_pm/notebooks/ch23_bridge_to_pytorch_llm.ipynb

Do not edit by hand — re-generate via tools/export_chapters_from_notebooks.py.
"""

# ---- [cell 1] ----------------------------------------
import numpy as np
np.set_printoptions(precision=3, suppress=True)

# ---- [cell 2] ----------------------------------------
def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-12)

# ---- [cell 3] ----------------------------------------
# Stabilized cross-entropy for a single row
rs = np.random.default_rng(23)
V = 8
z = rs.normal(size=(1, V))
y = rs.integers(0, V, size=1)
p = softmax(z)
ce = -np.log(p[0, y[0]])
print('CE=', float(ce))

# ---- [cell 4] ----------------------------------------
# Attention shape sanity
n, d, dh = 5, 6, 3
X = rs.normal(size=(n, d))
Wq = rs.normal(size=(d, dh)); Wk = rs.normal(size=(d, dh)); Wv = rs.normal(size=(d, dh))
Q, K, V = X @ Wq, X @ Wk, X @ Wv
S = (Q @ K.T) / np.sqrt(dh)
A = softmax(S)
print('shapes Q,K,V,A,O:', Q.shape, K.shape, V.shape, A.shape, (A @ V).shape)
