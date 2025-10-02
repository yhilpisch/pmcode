# Python & Mathematics for Data Science and Machine Learning
# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# AI-powered by GPT-5

"""Generated from notebook: /Users/yves/Library/CloudStorage/Dropbox/Program/books/4_pm/notebooks/ch08_differentiation_taylor_intuition.ipynb

Do not edit by hand — re-generate via tools/export_chapters_from_notebooks.py.
"""

# ---- [cell 1] ----------------------------------------
# [magics stripped] %config InlineBackend.figure_format = 'retina'
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')

# ---- [cell 2] ----------------------------------------
def f(x): return np.sin(x)
def df(x): return np.cos(x)
x0 = 1.0
for h in [1e-1, 1e-2, 1e-3]:
    fwd = (f(x0+h)-f(x0))/h
    cen = (f(x0+h)-f(x0-h))/(2*h)
    print(h, abs(fwd-df(x0)), abs(cen-df(x0)))

# ---- [cell 3] ----------------------------------------
def g(xy):
    x,y = xy; return x**2 + 3*x*y + 2*y**2
def grad_g(xy):
    x,y = xy; return np.array([2*x + 3*y, 3*x + 4*y], dtype=np.float64)
p = np.array([0.7, -0.2])
h = 1e-5; ex = np.array([h,0.0]); ey = np.array([0.0,h])
dgdx = (g(p+ex)-g(p-ex))/(2*h); dgdy=(g(p+ey)-g(p-ey))/(2*h)
print(np.allclose(np.array([dgdx,dgdy]), grad_g(p), atol=1e-6))

# ---- [cell 4] ----------------------------------------
def f2(xy):
    x,y = xy; return np.exp(x) + x*y + 0.5*y**2
def grad_f2(xy):
    x,y = xy; return np.array([np.exp(x) + y, x + y])
x0 = np.array([0.2, -0.3]); u = np.array([1.0, 2.0]); u /= np.linalg.norm(u)
h=1e-6; slope_fd = (f2(x0+h*u)-f2(x0-h*u))/(2*h); slope_true=grad_f2(x0).dot(u)
ok_dir = np.allclose(slope_fd, slope_true)
# Taylor errors (1st vs 2nd) for exp at x0
def f1(x): return np.exp(x)
def df1(x): return np.exp(x)
def d2f1(x): return np.exp(x)
x0s = [0.0, 0.5]; h=0.1
ratios=[]
for x0 in x0s:
    f0=f1(x0); t1=f0+df1(x0)*h; t2=f0+df1(x0)*h+0.5*d2f1(x0)*h**2
    err1=abs(f1(x0+h)-t1); err2=abs(f1(x0+h)-t2); ratios.append(err1/err2)
ok_dir, np.round(ratios,1)
