# Python & Mathematics for Data Science and Machine Learning
# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# AI-powered by GPT-5.x

# ---- [cell 1] ----------------------------------------
# Imports, plotting style, and reproducible RNG
# [magics stripped] %config InlineBackend.figure_format = 'retina'
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
# Fixed seed for reproducible random numbers across runs
rs = np.random.default_rng(42)

# ---- [cell 2] ----------------------------------------
def lln_demo(dist, n_values=(100, 1_000, 10_000)):
    """Check how the sample mean approaches the expectation.
    dist: small dict with 'mean' and 'gen'(rng,size)->array
    """
    mu = dist['mean']
    for n in n_values:
        x = dist['gen'](rs, size=n)
        m = x.mean()
        err = abs(m - mu)
        print(f"n={n:>6d}  mean={m:+.6f}  |mean-mu|={err:.6f}")
        # Sanity: finite outputs
        assert np.isfinite(m)

# Standard Normal: E[X] = 0
normal = {
    'mean': 0.0,
    'gen': lambda rng, size: rng.standard_normal(size=size).astype(np.float64),
}

lln_demo(normal)

# ---- [cell 3] ----------------------------------------
def variance_nonneg_demo(rng, n=50_000):
    # Draw samples and estimate E[X], E[X^2], and Var~ = E[X^2] - (E[X])^2
    x = rng.standard_normal(size=n).astype(np.float64)
    ex = x.mean()
    ex2 = (x * x).mean()
    var_est = ex2 - ex * ex
    print(f"E[X]={ex:+.4f}, E[X^2]={ex2:+.4f}, Var~={var_est:+.6f}")
    assert var_est > -1e-12  # allow tiny negatives from rounding

variance_nonneg_demo(rs)

# ---- [cell 4] ----------------------------------------
N = 20_000
x = rs.standard_normal(size=N).astype(np.float64)
running_mean = np.cumsum(x) / np.arange(1, N + 1)

fig, ax = plt.subplots(figsize=(6.8, 3.6), dpi=140)
ax.plot(np.arange(1, N + 1), running_mean, color='C0', lw=1.6, label='running mean')
ax.axhline(0.0, color='k', lw=1.0, ls='--', label=r'true mean $\mu=0$')
ax.set_xscale('log')
ax.set_xlabel('n (log scale)')
ax.set_ylabel(r'sample mean $\bar X_n$')
ax.set_title('LLN in action: running mean vs. sample size')
ax.legend(loc='best')
ax.grid(alpha=0.25)
plt.show()
