# Python & Mathematics for Data Science and Machine Learning
# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# AI-powered by GPT-5

"""Generated from notebook: /Users/yves/Library/CloudStorage/Dropbox/Program/books/4_pm/notebooks/ch18_linear_models_revisited_end_to_end.ipynb

Do not edit by hand — re-generate via tools/export_chapters_from_notebooks.py.
"""

# ---- [cell 1] ----------------------------------------
# [magics stripped] %config InlineBackend.figure_format = 'retina'
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
rs = np.random.default_rng(18)

# ---- [cell 2] ----------------------------------------
def standardize(X, y):
    mu = X.mean(axis=0, keepdims=True); sd = X.std(axis=0, keepdims=True)+1e-12
    return (X-mu)/sd, y-y.mean(), mu, sd
def ridge_fit(X, y, lam):
    d = X.shape[1]; I = np.eye(d);
    return np.linalg.solve(X.T@X + lam*I, X.T@y)
def soft(z,t): return np.sign(z)*np.maximum(np.abs(z)-t,0.0)
def lasso_cd(X,y,lam,iters=200,w0=None):
    n,d = X.shape; w = np.zeros(d) if w0 is None else w0.copy();
    L = (X**2).sum(axis=0)+1e-12
    for _ in range(iters):
        for j in range(d):
            r = y - (X@w) + X[:,j]*w[j]; rho = X[:,j].T@r
            w[j] = soft(rho/L[j], lam/L[j])
    return w
# synthetic correlated design
base = rs.normal(size=(200,1)); noise = rs.normal(scale=0.8,size=(200,7))
X = np.concatenate([base, 0.6*base + noise], axis=1)
w_true = np.array([2.0,-1.5,0.0,0.0,1.2,0.0,0.7,0.0])
y = X@w_true + rs.normal(scale=0.7, size=200)
Xs, ys, *_ = standardize(X,y)
lam_grid = np.logspace(-3,2,50); W_ridge=[]; W_lasso=[]; w=np.zeros(Xs.shape[1])
for lam in lam_grid: W_ridge.append(ridge_fit(Xs,ys,lam))
for lam in lam_grid[::-1]: w = lasso_cd(Xs,ys,lam,w0=w); W_lasso.append(w.copy())
W_ridge, W_lasso = np.stack(W_ridge), np.stack(W_lasso[::-1])
xs = np.log10(lam_grid); fig,axes=plt.subplots(1,2,figsize=(10,4))
[axes[0].plot(xs,W_ridge[:,j]) for j in range(W_ridge.shape[1])]; axes[0].set_title('Ridge paths'); axes[0].set_xlabel('log10(lambda)')
[axes[1].plot(xs,W_lasso[:,j]) for j in range(W_lasso.shape[1])]; axes[1].axhline(0,color='k',lw=0.5); axes[1].set_title('Lasso paths'); axes[1].set_xlabel('log10(lambda)')
plt.show()

# ---- [cell 3] ----------------------------------------
idx = rs.permutation(len(X)); tr=idx[:140]; va=idx[140:]
Xtr, ytr = X[tr], y[tr]; Xva,yva = X[va], y[va]
Xtr_s, ytr_s, Xva_s, yva_s = (Xtr - Xtr.mean(0))/ (Xtr.std(0)+1e-12), ytr-ytr.mean(), (Xva - Xtr.mean(0))/ (Xtr.std(0)+1e-12), yva-ytr.mean()
mses_tr, mses_va = [], []
for lam in lam_grid:
    w = ridge_fit(Xtr_s,ytr_s,lam); mses_tr.append(((Xtr_s@w - ytr_s)**2).mean()); mses_va.append(((Xva_s@w - yva_s)**2).mean())
plt.figure(figsize=(6,4)); plt.plot(np.log10(lam_grid),mses_tr,label='train'); plt.plot(np.log10(lam_grid),mses_va,label='val'); plt.xlabel('log10(lambda)'); plt.ylabel('MSE'); plt.legend(); plt.show()

# ---- [cell 4] ----------------------------------------
# OLS via pseudoinverse vs Ridge
w_ols = np.linalg.pinv(Xtr_s)@ytr_s; w_r = ridge_fit(Xtr_s,ytr_s,5.0)
r_ols = yva_s - Xva_s@w_ols; r_r = yva_s - Xva_s@w_r
fig,axes=plt.subplots(1,2,figsize=(10,4));
axes[0].scatter(Xva_s@w_ols,r_ols,s=10); axes[0].axhline(0,color='k',lw=0.5); axes[0].set_title('OLS residuals');
axes[1].scatter(Xva_s@w_r,r_r,s=10,color='tab:blue'); axes[1].axhline(0,color='k',lw=0.5); axes[1].set_title('Ridge residuals'); plt.show()
