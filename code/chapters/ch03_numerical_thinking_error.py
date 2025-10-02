# Python & Mathematics for Data Science and Machine Learning
# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# AI-powered by GPT-5

"""Generated from notebook: /Users/yves/Library/CloudStorage/Dropbox/Program/books/4_pm/notebooks/ch03_numerical_thinking_error.ipynb

Do not edit by hand — re-generate via tools/export_chapters_from_notebooks.py.
"""

# ---- [cell 1] ----------------------------------------
# [magics stripped] %config InlineBackend.figure_format = 'retina'
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
rs = np.random.default_rng(0)  # reproducible RNG

# ---- [cell 2] ----------------------------------------
def f_naive(x):
    return np.sqrt(x + 1.0) - np.sqrt(x)

def f_stable(x):
    # (a-b)(a+b)/(a+b) = (a^2-b^2)/(a+b) = 1/(sqrt{x+1}+sqrt{x})
    return 1.0 / (np.sqrt(x + 1.0) + np.sqrt(x))

xs = np.logspace(0, 12, 6, dtype=np.float64)
rel = np.abs((f_naive(xs) - f_stable(xs)) / f_stable(xs))
print(np.vstack([xs, rel]).T)

# ---- [cell 3] ----------------------------------------
A = np.array([[1., 1.], [1., 1.000001]])
b = np.array([2.0, 2.000001])
x = np.linalg.solve(A, b)
res = np.linalg.norm(A @ x - b)
k2 = np.linalg.cond(A)
print('x=', x)
print('residual=', res)
print('cond_2(A)=', k2)

# ---- [cell 4] ----------------------------------------
xs = np.logspace(6, 12, 4)
ratio = 2.0*np.sqrt(xs)*f_stable(xs)
np.round(ratio, 6)
