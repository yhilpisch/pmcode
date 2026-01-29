# Python & Mathematics for Data Science and Machine Learning
# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# AI-powered by GPT-5.x

# ---- [cell 1] ----------------------------------------
# [magics stripped] %config InlineBackend.figure_format = 'retina'
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext
plt.style.use('seaborn-v0_8')
rs = np.random.default_rng(42)
getcontext().prec = 60  # high precision for references

# ---- [cell 2] ----------------------------------------
# Machine epsilon (gap between 1 and next float64) and unit roundoff (half)
np.finfo(np.float64).eps, 0.5 * 2.0**-52

# ---- [cell 3] ----------------------------------------
0.1 + 0.2 == 0.3, Decimal('0.1') + Decimal('0.2') == Decimal('0.3')

# ---- [cell 4] ----------------------------------------
a = np.arange(6, dtype=np.float64).reshape(2, 3)
v = a[:, 1]           # view (slice)
c = a[:, [1]]         # copy (fancy index makes shape (2,1))
v[0] = 999.
a, v, c, np.shares_memory(a, v), np.shares_memory(a, c)

# ---- [cell 5] ----------------------------------------
A = np.array([[1., 2., 3.], [4., 5., 6.]])  # (2,3)
x = np.array([0.5, 1.0, -1.0])              # (3,)
y1 = A @ x
# Same result via einsum with explicit index roles: i = sum_j A_ij x_j
y2 = np.einsum('ij,j->i', A, x)
y1, y2, np.allclose(y1, y2)

# ---- [cell 6] ----------------------------------------
x, y, z = 1e16, 1.0, -1e16
(x + y) + z, x + (y + z)

# ---- [cell 7] ----------------------------------------
def kahan_sum(arr: np.ndarray) -> np.ndarray:
    s = 0.0
    c = 0.0
    out = np.empty_like(arr, dtype=np.float64)
    for i, x in enumerate(arr):
        y = x - c
        t = s + y
        c = (t - s) - y
        s = t
        out[i] = s
    return out

n = 20000
a = rs.uniform(0, 1e-10, size=n).astype(np.float64)

# High-precision reference prefix sums
pref = []
acc = Decimal(0)
for x_ in a:
    acc += Decimal(str(x_))  # preserve decimal digits
    pref.append(acc)
ref = np.array([float(v) for v in pref], dtype=np.float64)

naive = np.cumsum(a)
kahan = kahan_sum(a)

err_naive = np.abs(naive - ref)
err_kahan = np.abs(kahan - ref)

fig, ax = plt.subplots(figsize=(6.8, 3.6), dpi=140)
x_ix = np.arange(1, n + 1)
ax.plot(x_ix, err_naive, label='naive cumsum', color='C1', lw=1.4)
ax.plot(x_ix, err_kahan, label='Kahan sum', color='C2', lw=1.4)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('n (log scale)')
ax.set_ylabel('absolute error (log scale)')
ax.set_title('Rounding error in summation: naive vs Kahan')
ax.legend()
ax.grid(alpha=0.25)
plt.show()

# ---- [cell 8] ----------------------------------------
import math
import numpy as np
def softplus_naive(x):
    return np.log1p(np.exp(x))
def softplus_stable(x):  # numerically stable implementation
    x = np.asarray(x, dtype=np.float64)
    pos = x > 0
    y = np.empty_like(x)
    y[pos]  = x[pos] + np.log1p(np.exp(-x[pos]))
    y[~pos] = np.log1p(np.exp(x[~pos]))
    return y
xs = np.array([-1000., -50., -1., 0., 1., 50., 1000.])
np.exp(1000), softplus_naive(xs), softplus_stable(xs)

# ---- [cell 9] ----------------------------------------
a = np.array([1000., 999., 995.])
m = a.max()
naive = np.log(np.sum(np.exp(a)))
stable = m + np.log(np.sum(np.exp(a - m)))
naive, stable

# ---- [cell 10] ----------------------------------------
import math
xs = [0., 1., 2., 3.]
loop_exp = [math.exp(t) for t in xs]
vec_exp = np.exp(np.array(xs))
loop_exp, vec_exp

# ---- [cell 11] ----------------------------------------
X = np.array([[0., 0.], [1., 0.], [0., 1.]])
diff = X[:, None, :] - X[None, :, :]
D = np.sqrt(np.sum(diff**2, axis=-1))
D

# ---- [cell 12] ----------------------------------------
rng = np.random.default_rng(7)
x = rng.standard_normal(size=50_000).astype(np.float64)
fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=140)
counts, bins, _ = ax.hist(
    x,
    bins=80,
    density=True,
    alpha=0.35,
    color='C0',
    label='samples (hist)',
)
grid = np.linspace(bins[0], bins[-1], 600)
pdf = (1.0/np.sqrt(2*np.pi))*np.exp(-0.5*grid*grid)
ax.plot(grid, pdf, color='C1', lw=2.0, label='analytic PDF')
ax.set_xlabel('x')
ax.set_ylabel('density')
ax.set_title('Standard Normal: histogram and analytic PDF')
ax.legend(); ax.grid(alpha=0.25)
plt.show()

# ---- [cell 13] ----------------------------------------
xs = np.linspace(-3, 3, 200)
ys = np.linspace(-3, 3, 200)
Xg, Yg = np.meshgrid(xs, ys)
Z = np.exp(-(Xg**2 + Yg**2))
fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=140)
im = ax.imshow(
    Z,
    extent=[xs.min(), xs.max(), ys.min(), ys.max()],
    origin='lower',
    cmap='viridis',
    aspect='auto',
)
fig.colorbar(im, ax=ax, shrink=0.85)
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('Gaussian bump: z = exp(-(x^2 + y^2))')
plt.show()

# ---- [cell 14] ----------------------------------------
x, y, z = 1e16, 1.0, -1e16
(x + y) + z, x + (y + z)
