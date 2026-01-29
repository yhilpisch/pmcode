# Python & Mathematics for Data Science and Machine Learning
# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# AI-powered by GPT-5.x

# ---- [cell 1] ----------------------------------------
# [magics stripped] %config InlineBackend.figure_format = 'retina'
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8')

# ---- [cell 2] ----------------------------------------
rng = np.random.default_rng(0)  # reproducible parameters
W1 = rng.normal(size=(4, 3)).astype(np.float64)
b1 = rng.normal(size=4).astype(np.float64)
W2 = rng.normal(size=(2, 4)).astype(np.float64)
b2 = rng.normal(size=2).astype(np.float64)
x = rng.normal(size=3).astype(np.float64)
target = rng.normal(size=2).astype(np.float64)

def forward(W1, b1, W2, b2, x):
    """Two-layer network with tanh hidden and L2 loss."""
    a1 = W1 @ x + b1
    h = np.tanh(a1)
    z = W2 @ h + b2
    residual = z - target
    loss = 0.5 * np.sum(residual**2)
    return loss, (a1, h, z, residual)

def backward(W1, b1, W2, b2, x, cache):
    """Manual backprop that mirrors the algebra in the chapter."""
    a1, h, z, residual = cache
    grad_z = residual  # derivative of 0.5||z-target||^2
    grad_W2 = grad_z[:, None] * h[None, :]
    grad_b2 = grad_z
    grad_h = W2.T @ grad_z
    grad_a1 = grad_h * (1.0 - h**2)  # tanh' = 1 - tanh^2
    grad_W1 = grad_a1[:, None] * x[None, :]
    grad_b1 = grad_a1
    grad_x = W1.T @ grad_a1
    return grad_W1, grad_b1, grad_W2, grad_b2, grad_x

# ---- [cell 3] ----------------------------------------
loss, cache = forward(W1, b1, W2, b2, x)
grads = backward(W1, b1, W2, b2, x, cache)

print(f'loss: {loss:.4f}')
for label, grad in zip(['dW1', 'db1', 'dW2', 'db2', 'dx'], grads):
    print(f'{label} shape: {grad.shape}')

# ---- [cell 4] ----------------------------------------
def finite_diff(param, loss_fn, eps=1e-5):
    """Central-difference gradient for every entry in param."""
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        original = param[idx]
        param[idx] = original + eps
        loss_plus, _ = loss_fn()
        param[idx] = original - eps
        loss_minus, _ = loss_fn()
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        param[idx] = original
        it.iternext()
    return grad

def loss_only():
    """Closure that recomputes the loss with current parameters."""
    return forward(W1, b1, W2, b2, x)

checks = [
    np.allclose(finite_diff(W1, loss_only), grads[0], atol=1e-6),
    np.allclose(finite_diff(b1, loss_only), grads[1], atol=1e-6),
    np.allclose(finite_diff(W2, loss_only), grads[2], atol=1e-6),
    np.allclose(finite_diff(b2, loss_only), grads[3], atol=1e-6),
    np.allclose(finite_diff(x, loss_only), grads[4], atol=1e-6),
]
print('All finite-difference checks pass:', all(checks))

# ---- [cell 5] ----------------------------------------
rng = np.random.default_rng(1)
W = rng.normal(size=(4, 3)).astype(np.float64)
x = rng.normal(size=3).astype(np.float64)
v = rng.normal(size=3).astype(np.float64)  # forward-mode seed
u = rng.normal(size=4).astype(np.float64)  # reverse-mode seed

def f(x):
    """Tanh layer used in the chapter's JVP/VJP discussion."""
    return np.tanh(W @ x)

def jvp_exact(x, v):
    """Analytic JVP = J(x) @ v."""
    preact = W @ x
    diag = 1.0 - np.tanh(preact) ** 2
    J = diag[:, None] * W
    return J @ v

def vjp_exact(x, u):
    """Analytic VJP = u^T J(x)."""
    preact = W @ x
    diag = 1.0 - np.tanh(preact) ** 2
    J = diag[:, None] * W
    return u @ J

# ---- [cell 6] ----------------------------------------
eps = 1e-5  # central-difference step
jvp_fd = (f(x + eps * v) - f(x - eps * v)) / (2 * eps)

phi = lambda z: np.dot(u, f(z))  # scalar projection for the VJP check
vjp_fd = np.zeros_like(x)
for i in range(x.size):  # sweep standard basis directions
    e_i = np.zeros_like(x)
    e_i[i] = eps
    vjp_fd[i] = (phi(x + e_i) - phi(x - e_i)) / (2 * eps)

print('JVP matches:', np.allclose(jvp_fd, jvp_exact(x, v), atol=1e-7))
print('VJP matches:', np.allclose(vjp_fd, vjp_exact(x, u), atol=1e-7))
