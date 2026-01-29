# Python & Mathematics for Data Science and Machine Learning
# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# AI-powered by GPT-5.x

# ---- [cell 1] ----------------------------------------
# [magics stripped] %config InlineBackend.figure_format = 'retina'
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.style.use('seaborn-v0_8')
rs = np.random.default_rng(13)

# ---- [cell 2] ----------------------------------------
def beta_posterior(alpha: float, beta: float, k: int, n: int):
    """Return posterior parameters (alpha', beta') for Beta-Bernoulli.
    Prior: Beta(alpha, beta). Data: k successes in n trials.
    """
    return alpha + k, beta + (n - k)

p_true = 0.3
alpha0, beta0 = 2.0, 2.0
n = 100
x = rs.binomial(1, p_true, size=n).astype(int)
k = int(x.sum())
a1, b1 = beta_posterior(alpha0, beta0, k, n)
print(f'k={k}, posterior=Beta({a1},{b1})')
assert 0 <= k <= n

# ---- [cell 3] ----------------------------------------
# Quick plot: prior vs posterior density
import math
def log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

def beta_pdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
    x = np.clip(x, 1e-12, 1 - 1e-12)
    lb = log_beta(a, b)
    return np.exp((a-1)*np.log(x) + (b-1)*np.log(1-x) - lb)

xs = np.linspace(0.0, 1.0, 500)
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.plot(
    xs,
    beta_pdf(xs, alpha0, beta0),
    lw=1.8,
    label=f'Prior Beta({alpha0:.0f},{beta0:.0f})',
)
ax.plot(xs, beta_pdf(xs, a1, b1), lw=2.2, label=f'Posterior Beta({a1},{b1})')
ax.axvline(p_true, ls='--', color='#e84855', lw=1.4, label='p_true')
ax.set_xlabel('p')
ax.set_ylabel('density')
ax.legend(loc='best')
ax.set_title('Beta–Bernoulli: prior vs posterior')
fig.tight_layout()
plt.show()

# ---- [cell 4] ----------------------------------------
def H_bernoulli(p):
    p = np.clip(p, 1e-12, 1-1e-12)
    return -(p*np.log(p) + (1-p)*np.log(1-p))

def KL_bernoulli(p, q):
    p = np.clip(p, 1e-12, 1-1e-12)
    q = np.clip(q, 1e-12, 1-1e-12)
    return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

p = np.linspace(0.0, 1.0, 400)
fig, ax = plt.subplots(figsize=(6.0, 3.2))
ax.plot(p, H_bernoulli(p), lw=2.0)
ax.set_title('Entropy H(p) for Bernoulli')
ax.set_xlabel('p')
ax.set_ylabel('nats')
ax.axvline(0.5, ls='--', color='#888', lw=1.0)
fig.tight_layout()
plt.show()

# Small KL table for intuition
pairs = [(0.1, 0.2), (0.1, 0.9), (0.3, 0.28), (0.9, 0.1)]
for pp, qq in pairs:
    kl_val = KL_bernoulli(np.array([pp]), np.array([qq]))[0]
    print(f"D_KL(Ber({pp}) || Ber({qq})) = {kl_val:.4f} nats")

# ---- [cell 5] ----------------------------------------
def cross_entropy_empirical(y, q):
    q = np.clip(q, 1e-12, 1-1e-12)
    return np.mean(-(y*np.log(q) + (1-y)*np.log(1-q)))

def cross_entropy_param(p, q):
    p = np.clip(p, 1e-12, 1-1e-12)
    q = np.clip(q, 1e-12, 1-1e-12)
    return -(p*np.log(q) + (1-p)*np.log(1-q))

p_true, q_model, n = 0.3, 0.28, 20000
y = rs.binomial(1, p_true, size=n)
nll = cross_entropy_empirical(y, q_model)
xent = cross_entropy_param(y.mean(), q_model)
print(f'mean NLL ~ {nll:.5f}, cross-entropy(p_hat,q) ~ {xent:.5f}')
assert abs(nll - xent) < 5e-3
