# Python & Mathematics for Data Science and Machine Learning
# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# AI-powered by GPT-5

"""Generated from notebook: /Users/yves/Library/CloudStorage/Dropbox/Program/books/4_pm/notebooks/ch16_constrained_convex_optimization.ipynb

Do not edit by hand — re-generate via tools/export_chapters_from_notebooks.py.
"""

# ---- [cell 1] ----------------------------------------
# [magics stripped] %config InlineBackend.figure_format = 'retina'
import numpy as np  # arrays
import matplotlib.pyplot as plt  # plotting
plt.style.use('seaborn-v0_8')  # house style

# ---- [cell 2] ----------------------------------------
def grad(x, Q, c):  # ∇(0.5 x^T Q x + c^T x)
    return Q @ x + c

def proj_box(x, lo, hi):  # clip to [lo,hi] per-coordinate
    return np.minimum(np.maximum(x, lo), hi)

Q = np.array([[6.0, 2.0], [2.0, 1.0]])  # SPD bowl
c = np.array([-3.0, -1.0])  # linear term
lo, hi = np.array([-1.0, 0.0]), np.array([2.0, 2.0])  # box
x0 = np.array([1.6, 1.6])  # start
alpha, steps = 0.3, 12  # step and steps

def run_pgd(x):  # projected gradient steps
    xs = [x.copy()]
    for _ in range(steps):
        x = proj_box(x - alpha*grad(x, Q, c), lo, hi)
        xs.append(x.copy())
    return np.array(xs)

pgd = run_pgd(x0)  # path
xs = np.linspace(-1.5, 2.5, 240); ys = np.linspace(-0.5, 2.5, 240)  # grid
X, Y = np.meshgrid(xs, ys)
Z = 0.5*(Q[0,0]*X**2 + 2*Q[0,1]*X*Y + Q[1,1]*Y**2) + c[0]*X + c[1]*Y
fig, ax = plt.subplots(figsize=(5.6,4.4))
ax.contour(X, Y, Z, levels=12, cmap='viridis_r')  # contours
ax.plot([lo[0],hi[0],hi[0],lo[0],lo[0]],[lo[1],lo[1],hi[1],hi[1],lo[1]],'-',color='#555')  # box
ax.plot(pgd[:,0], pgd[:,1], 'o-', color='#1b998b', ms=3)  # path
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$'); ax.set_aspect('equal','box')
plt.tight_layout(); plt.show()

# ---- [cell 3] ----------------------------------------
Q = np.array([[4.0, 1.0],[1.0, 1.5]]); c = np.array([-2.0,-0.5])  # objective
A = np.array([[1.0, 1.0]]); b = np.array([1.0])  # equality a^T x = b
# Solve block KKT system [Q A^T; A 0][x;λ]=[-c;b]
K = np.block([[Q, A.T],[A, np.zeros((1,1))]])
rhs = np.concatenate([-c, b])
sol = np.linalg.solve(K, rhs); xstar, lam = sol[:2], sol[2]  # solution
# Visualize tangency
xs = np.linspace(-0.2, 1.4, 220); ys = np.linspace(-0.2, 1.4, 220)
X, Y = np.meshgrid(xs, ys)
Z = 0.5*(Q[0,0]*X**2 + 2*Q[0,1]*X*Y + Q[1,1]*Y**2) + c[0]*X + c[1]*Y
fig, ax = plt.subplots(figsize=(5.2,4.4))
ax.contour(X, Y, Z, levels=12, cmap='viridis_r', alpha=0.8)
fstar = 0.5*float(xstar @ Q @ xstar) + float(c @ xstar)
ax.contour(X, Y, Z, levels=[fstar], colors=['#333'], linewidths=2.0)  # tangent level
xx = np.linspace(xs.min(), xs.max(), 400); yy = (b[0] - xx)  # x1+x2=1
mask = (yy>=ys.min()) & (yy<=ys.max()); ax.plot(xx[mask], yy[mask], '-', color='#555')
ax.plot([xstar[0]],[xstar[1]],'o',color='#111',ms=6)  # solution
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$'); ax.set_aspect('equal','box')
plt.tight_layout(); plt.show()
print('x*=', xstar, ' lambda=', float(lam))

# ---- [cell 4] ----------------------------------------
# Solve: min 0.5 x^T Q x + c^T x  s.t.  0.5||x||^2 <= 0.5 R^2
Q = np.array([[6.0, 2.0], [2.0, 1.0]]); c = np.array([-3.0, -1.0]); R = 0.4
I = np.eye(2)
# Unconstrained minimizer
x_unc = -np.linalg.solve(Q, c)
print('||x_unc||=', np.linalg.norm(x_unc))
# Bisection on λ to enforce ||x(λ)|| = R when boundary active
if np.linalg.norm(x_unc) <= R + 1e-12:
    x_star, lam = x_unc, 0.0
else:
    def norm_minus_R(lmb):
        x = -np.linalg.solve(Q + lmb*I, c)
        return np.linalg.norm(x) - R
    lo, hi = 0.0, 1.0
    while norm_minus_R(hi) > 0.0 and hi < 1e6: hi *= 2.0
    for _ in range(60):
        mid = 0.5*(lo+hi)
        if norm_minus_R(mid) > 0.0: lo = mid
        else: hi = mid
    lam = 0.5*(lo+hi)
    x_star = -np.linalg.solve(Q + lam*I, c)
# KKT checks
g = Q @ x_star + c  # ∇f(x*)
phi = 0.5*(x_star @ x_star - R*R)  # constraint value
stat = np.linalg.norm(g + lam * x_star)  # stationarity residual
print('x*=', x_star, ' lambda=', lam)
print('stationarity=', f'{stat:.2e}', ' primal=', f'{phi:.2e}', ' dual=', f'{lam:.2e}', ' comp=', f'{(lam*phi):.2e}')
