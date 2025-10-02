# Python & Mathematics for Data Science and Machine Learning
# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# AI-powered by GPT-5

"""Generated from notebook: /Users/yves/Library/CloudStorage/Dropbox/Program/books/4_pm/notebooks/ch05_matrices_linear_maps_bases.ipynb

Do not edit by hand — re-generate via tools/export_chapters_from_notebooks.py.
"""

# ---- [cell 1] ----------------------------------------
# [magics stripped] %config InlineBackend.figure_format = 'retina'
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
rs = np.random.default_rng(0)

# ---- [cell 2] ----------------------------------------
A = np.array([[1., 1.],[1., 1.000001],[0., 0.000001]])
b = np.array([2.0, 2.000001, 0.000001])
print('rank(A)=', np.linalg.matrix_rank(A))
x_qr, *_ = np.linalg.lstsq(A, b, rcond=None)
x_ne = np.linalg.solve(A.T @ A, A.T @ b)
res_qr = np.linalg.norm(A @ x_qr - b)
res_ne = np.linalg.norm(A @ x_ne - b)
print('res_qr=', res_qr, ' res_ne=', res_ne)

# ---- [cell 3] ----------------------------------------
ATA = A.T @ A
P = A @ np.linalg.inv(ATA) @ A.T
print(np.allclose(P @ P, P), np.allclose(P.T, P))

# ---- [cell 4] ----------------------------------------
rank = np.linalg.matrix_rank(A)
_, S, Vt = np.linalg.svd(A, full_matrices=False)
null_dim = np.sum(S < 1e-12)
rank + null_dim, A.shape[1]
