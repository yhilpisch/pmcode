# Python & Mathematics for Data Science and Machine Learning
# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# AI-powered by GPT-5

"""Generated from notebook: /Users/yves/Library/CloudStorage/Dropbox/Program/books/4_pm/notebooks/ch15_unconstrained_optimization.ipynb

Do not edit by hand — re-generate via tools/export_chapters_from_notebooks.py.
"""

# ---- [cell 1] ----------------------------------------
# [magics stripped] %config InlineBackend.figure_format = 'retina'
import numpy as np  # arrays, RNG
import matplotlib.pyplot as plt  # plotting
plt.style.use('seaborn-v0_8')  # house style
rng = np.random.default_rng(15)  # seed

# ---- [cell 2] ----------------------------------------
def make_Q(L=50.0, m=1.0, theta_deg=30.0):  # SPD with rotation
    t = np.deg2rad(theta_deg)  # radians
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])  # rotation
    return R.T @ np.diag([L, m]) @ R  # rotated diag

def f_quad(x, Q):  # 0.5 x^T Q x
    return 0.5 * float(x @ Q @ x)

def g_quad(x, Q):  # ∇f = Q x
    return Q @ x

def backtrack(x, Q, alpha0=0.2, beta=0.5, c=1e-4):  # Armijo backtracking
    g = g_quad(x, Q); t = alpha0; fx = f_quad(x, Q); gg = float(g @ g)
    while f_quad(x - t*g, Q) > fx - c*t*gg:  # sufficient decrease
        t *= beta  # shrink
    return t  # accepted step

def gd_path(Q, x0, steps, alpha=None):  # fixed step or backtracking
    x = x0.astype(float).copy(); xs = [x.copy()]
    for _ in range(steps):
        g = g_quad(x, Q)  # gradient
        t = alpha if alpha is not None else backtrack(x, Q)  # step size
        x = x - t*g  # update
        xs.append(x.copy())  # record
    return np.array(xs)

Q = make_Q(); x0 = np.array([2.0, -1.0])  # test bowl
p_fix = gd_path(Q, x0, steps=28, alpha=0.03)  # fixed step
p_bt = gd_path(Q, x0, steps=28, alpha=None)  # backtracking

# contours covering both paths
xs = np.linspace(min(p_fix[:,0].min(), p_bt[:,0].min())-0.5,
                max(p_fix[:,0].max(), p_bt[:,0].max())+0.5, 220)
ys = np.linspace(min(p_fix[:,1].min(), p_bt[:,1].min())-0.5,
                max(p_fix[:,1].max(), p_bt[:,1].max())+0.5, 220)
X, Y = np.meshgrid(xs, ys)
Z = 0.5*(Q[0,0]*X**2 + 2*Q[0,1]*X*Y + Q[1,1]*Y**2)

fig, ax = plt.subplots(1,2, figsize=(10,4), sharex=True, sharey=True)
for a, path, col, title in [(ax[0], p_fix, '#e84855', 'Fixed step'), (ax[1], p_bt, '#1b998b', 'Backtracking')]:
    a.contour(X, Y, Z, levels=12, cmap='viridis')  # contours
    a.plot(path[:,0], path[:,1], 'o-', color=col, ms=3)  # path
    a.plot([x0[0]], [x0[1]], 'o', color='k', ms=6)  # start
    a.set_title(title)  # title
    a.set_xlabel('$x_1$')  # label
ax[0].set_ylabel('$x_2$')  # y label
plt.tight_layout(); plt.show()  # render

# ---- [cell 3] ----------------------------------------
# Quadratic with known spectrum: L=10, μ=1
L, mu = 10.0, 1.0  # Lipschitz and strong convexity
Q = np.diag([L, mu])  # SPD
f = lambda v: 0.5*float(v @ Q @ v)  # objective
g = lambda v: Q @ v  # gradient
alpha = 0.9 / L  # safe step (<2/L)
# Descent-lemma bound and monotone decrease
x = np.array([2.0, -1.5], float); ok=True; vals=[]
for _ in range(40):
    gv = g(x); fx = f(x)
    x_next = x - alpha * gv; fx_next = f(x_next)
    bound = fx - alpha*(1 - 0.5*L*alpha)*float(gv @ gv)
    ok &= (fx_next <= bound + 1e-12)  # bound holds
    vals.append(fx_next); x = x_next
mono = all(vals[i+1] <= vals[i] + 1e-12 for i in range(len(vals)-1))
# Linear contraction vs theory ρ
rho = max(abs(1 - alpha*mu), abs(1 - alpha*L))
x = np.array([2.0, -1.5], float); ratios=[]
for _ in range(40):
    x_prev = x.copy(); x = x - alpha * g(x)
    if (np.linalg.norm(x_prev)>0): ratios.append(np.linalg.norm(x)/np.linalg.norm(x_prev))
emp = max(ratios) if ratios else 0.0
print('Descent lemma:', ok, '| Monotone:', mono, '| Emp ≤ rho:', emp <= rho + 1e-12)
print(f"rho_theory={rho:.3f}  emp_max={emp:.3f}")

# ---- [cell 4] ----------------------------------------
def func_1d(x):  # quartic slice
    return 0.1*x**4 - 1.5*x**2

def grad_1d(x):  # derivative
    return 0.4*x**3 - 3.0*x

x0 = 4.0  # start
g0 = grad_1d(x0); p = -g0  # downhill
t = np.linspace(0, 1.4, 400)  # steps
phi = func_1d(x0 + t*p); phi0 = func_1d(x0)  # loss
c, beta = 0.2, 0.7  # Armijo params
armijo = phi0 + c*t*p*g0  # linear bound

fig, ax = plt.subplots(1,2, figsize=(10,4))  # panels
ax[0].plot(t, phi, label=r'$\phi(t)$')  # loss
ax[0].plot(t, armijo, 'r--', label='Armijo bound')  # bound
# simple backtracking marks
tt = 1.0
for _ in range(5):
    if func_1d(x0 + tt*p) <= phi0 + c*tt*p*g0:  # accepted
        ax[0].plot([tt], [func_1d(x0+tt*p)], 'o', color='#1b998b'); break
    ax[0].plot([tt], [func_1d(x0+tt*p)], 'x', color='#e84855'); tt *= beta
ax[0].legend(); ax[0].set_xlabel('t'); ax[0].set_ylabel(r'$\phi(t)$')

ax[1].plot(t, phi, label='loss')  # loss
c_strict, c_len = 0.5, 0.05  # two c
ax[1].plot(t, phi0 + c_strict*t*p*g0, 'm--', label='strict')
ax[1].plot(t, phi0 + c_len*t*p*g0, 'c--', label='lenient')
ax[1].legend(); ax[1].set_xlabel('t')
plt.tight_layout(); plt.show()
