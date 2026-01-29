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
def norms(x):
    x = np.asarray(x, dtype=np.float64)
    l1 = np.sum(np.abs(x))
    l2 = np.sqrt(np.sum(x*x))
    linf = np.max(np.abs(x))
    return l1, l2, linf

x = rs.normal(size=5)
l1, l2, li = norms(x)
print(li <= l2 <= l1, l1 <= np.sqrt(x.size)*l2)

# ---- [cell 3] ----------------------------------------
def cosine_similarity(x, y):
    xn = x/np.linalg.norm(x); yn = y/np.linalg.norm(y)
    return float(xn @ yn)

a, b = rs.normal(size=3), rs.normal(size=3)
print(round(cosine_similarity(a,b), 3))

# ---- [cell 4] ----------------------------------------
def l2_from_cos(x, y):
    xn = x/np.linalg.norm(x); yn = y/np.linalg.norm(y)
    d2 = np.sum((xn-yn)**2); c = float(xn @ yn)
    return d2, 2.0*(1.0-c)
ok=[]
for _ in range(3):
    a,b = rs.normal(size=5), rs.normal(size=5)
    d2,rhs = l2_from_cos(a,b)
    ok.append(np.allclose(d2, rhs))
ok
