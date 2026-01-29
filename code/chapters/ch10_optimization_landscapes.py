# Python & Mathematics for Data Science and Machine Learning
# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# AI-powered by GPT-5.x

# ---- [cell 1] ----------------------------------------

# [magics stripped] %config InlineBackend.figure_format = 'retina'
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8')

# ---- [cell 2] ----------------------------------------

def f_ridge(xy):
    x, y = xy
    return 0.5 * (10 * x**2 + y**2)

def grad_ridge(xy):
    x, y = xy
    return np.array([10 * x, y], dtype=np.float64)

def hess_ridge(_xy):
    return np.array([[10.0, 0.0], [0.0, 1.0]])

def f_saddle(xy):
    x, y = xy
    return x**2 - y**2

def grad_saddle(xy):
    x, y = xy
    return np.array([2 * x, -2 * y], dtype=np.float64)

def hess_saddle(_xy):
    return np.array([[2.0, 0.0], [0.0, -2.0]])

for name, grad, hess in [
    ("ridge", grad_ridge, hess_ridge),
    ("saddle", grad_saddle, hess_saddle),
]:
    grad0 = grad((0.0, 0.0))
    eigs = np.linalg.eigvals(hess((0.0, 0.0)))
    eigs_rounded = np.round(eigs, 2)
    print(
        f"{name:>6}: ||grad||={np.linalg.norm(grad0):.1f} "
        f"eigs={eigs_rounded}"
    )

# ---- [cell 3] ----------------------------------------

grid = np.linspace(-2.0, 2.0, 200)
X, Y = np.meshgrid(grid, grid)
Z_ridge = f_ridge((X, Y))
Z_saddle = f_saddle((X, Y))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
cs = axes[0].contour(X, Y, Z_ridge, levels=18, cmap='Blues')
axes[0].clabel(cs, fmt='%.1f', fontsize=8, inline=True)
gx_r, gy_r = grad_ridge((X, Y))
axes[0].quiver(X[::6, ::6], Y[::6, ::6], -gx_r[::6, ::6], -gy_r[::6, ::6],
               color='#1b998b', alpha=0.85, scale=26)
axes[0].set_title('Convex bowl')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_aspect('equal', 'box')

cs = axes[1].contour(X, Y, Z_saddle, levels=18, cmap='RdBu')
axes[1].clabel(cs, fmt='%.1f', fontsize=8, inline=True)
gx_s, gy_s = grad_saddle((X, Y))
axes[1].quiver(X[::6, ::6], Y[::6, ::6], -gx_s[::6, ::6], -gy_s[::6, ::6],
               color='#e84855', alpha=0.85, scale=35)
axes[1].set_title('Saddle')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_aspect('equal', 'box')
plt.show()

# ---- [cell 4] ----------------------------------------

def gradient_descent(grad, x0, eta, steps):
    xs = [np.array(x0, dtype=np.float64)]
    for _ in range(steps):
        xs.append(xs[-1] - eta * grad(xs[-1]))
    return np.stack(xs)

start = np.array([1.0, 1.0])
slow = gradient_descent(grad_ridge, start, eta=0.08, steps=60)
fast = gradient_descent(grad_ridge, start, eta=0.18, steps=60)

levels = np.linspace(0.0, 6.0, 25)
xs = np.linspace(-1.0, 1.0, 200)
ys = np.linspace(-1.0, 1.0, 200)
X, Y = np.meshgrid(xs, ys)
Z = f_ridge((X, Y))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
axes[0].contour(X, Y, Z, levels=levels, cmap='Blues')
axes[0].plot(slow[:, 0], slow[:, 1], 'o-', ms=3, label='eta=0.08')
axes[0].plot(fast[:, 0], fast[:, 1], 'o-', ms=3, label='eta=0.18')
axes[0].set_title('Trajectories')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].legend(loc='upper right')
axes[0].set_aspect('equal', 'box')

for series, label in [(slow, 'eta=0.08'), (fast, 'eta=0.18')]:
    residual = np.linalg.norm(series - series[-1], axis=1)
    axes[1].semilogy(residual, label=label)
axes[1].set_title('Residual norm vs iteration')
axes[1].set_xlabel('iteration')
axes[1].set_ylabel('||x_k - x_*||_2')
axes[1].legend(loc='upper right')
fig.tight_layout()
plt.show()

# ---- [cell 5] ----------------------------------------

def armijo_ok(f, xk, pk, grad_fk, eta, c1=1e-4):
    return f(xk + eta * pk) <= f(xk) + c1 * eta * grad_fk @ pk

xk = np.array([0.8, 0.8])
pk = -grad_ridge(xk)
result = {
    eta: armijo_ok(f_ridge, xk, pk, grad_ridge(xk), eta)
    for eta in (0.05, 0.2, 0.35)
}
print(result)
