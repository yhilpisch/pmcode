# Python & Mathematics for Data Science and Machine Learning
# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# AI-powered by GPT-5.x

# ---- [cell 1] ----------------------------------------
# [magics stripped] %config InlineBackend.figure_format = 'retina'
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
rs = np.random.default_rng(14)

# ---- [cell 2] ----------------------------------------
def make_regression(n=140, noise=0.2, seed=0):
    r = np.random.default_rng(seed)
    x = np.linspace(-1, 1, n)
    y_true = np.sin(3*x)
    y = y_true + noise * r.standard_normal(n)
    return x, y, y_true

def poly_features(x, deg):
    return np.vstack([x**k for k in range(deg+1)]).T

def fit_ls(X, y):
    return np.linalg.pinv(X) @ y

x, y, y_true = make_regression()
idx = rs.permutation(len(x))
tr = idx[:90]; te = idx[90:]
def mse(a,b): return float(np.mean((a-b)**2))
train_mse, test_mse = [], []
for d in range(0, 16):
    X = poly_features(x, d)
    w = fit_ls(X[tr], y[tr])
    train_mse.append(mse(X[tr]@w, y[tr]))
    test_mse.append(mse(X[te]@w, y[te]))
deg_min = int(np.argmin(test_mse))
print('min test MSE at degree', deg_min)
fig, ax = plt.subplots(figsize=(6.0,3.6))
ax.plot(range(16), train_mse, '-o', label='train MSE')
ax.plot(range(16), test_mse, '-o', label='test MSE')
ax.set_xlabel('degree')
ax.set_ylabel('MSE')
ax.legend()
fig.tight_layout()
plt.show()

# ---- [cell 3] ----------------------------------------
def ridge(X, y, lam):
    I = np.eye(X.shape[1])
    return np.linalg.solve(X.T@X + lam*I, X.T@y)

x = np.linspace(-1, 1, 140)
X = np.vstack([x**k for k in range(11)]).T
y = np.sin(3*x) + 0.25*np.random.default_rng(1).standard_normal(140)
lams = np.logspace(-6, 1, 40)
idx_sel = [1,3,5,7,10]
tracks = []
for lam in lams:
    w = ridge(X, y, lam)
    tracks.append(w[idx_sel])
tracks = np.array(tracks)
fig, ax = plt.subplots(figsize=(6.4,3.6))
for j, idx in enumerate(idx_sel):
    ax.plot(lams, tracks[:,j], label=f'w[{idx}]')
ax.set_xscale('log')
ax.set_xlabel('lambda (log)')
ax.set_ylabel('coeff')
ax.legend()
fig.tight_layout()
plt.show()

# ---- [cell 4] ----------------------------------------
n, k = 40, 5
idx = np.arange(n)
folds = np.array_split(idx, k)
A = np.zeros((n, k), dtype=float)
for j, f in enumerate(folds):
    A[f, j] = 1.0
fig, ax = plt.subplots(figsize=(5.5,4.4))
im = ax.imshow(A, cmap='Greens', origin='lower', aspect='auto', vmin=0.0, vmax=1.0)
ax.set_xlabel('fold index'); ax.set_ylabel('sample index')
fig.colorbar(im, ax=ax, ticks=[0,1]).ax.set_yticklabels(['train','test'])
fig.tight_layout(); plt.show()
