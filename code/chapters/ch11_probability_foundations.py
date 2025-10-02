# Python & Mathematics for Data Science and Machine Learning
# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# AI-powered by GPT-5

"""Generated from notebook: /Users/yves/Library/CloudStorage/Dropbox/Program/books/4_pm/notebooks/ch11_probability_foundations.ipynb

Do not edit by hand — re-generate via tools/export_chapters_from_notebooks.py.
"""

# ---- [cell 1] ----------------------------------------
# [magics stripped] %config InlineBackend.figure_format = 'retina'
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.style.use('seaborn-v0_8')
rng = np.random.default_rng(0)

# ---- [cell 2] ----------------------------------------
def sample_bernoulli(p: float, size: int, *, rng: np.random.Generator) -> np.ndarray:
    return rng.binomial(1, p, size=size).astype(np.float64)

def monte_carlo_mean(samples: np.ndarray) -> tuple[float, float]:
    mean = samples.mean()
    stderr = samples.std(ddof=1) / np.sqrt(len(samples))
    return float(mean), float(stderr)

bernoulli_draws = sample_bernoulli(0.3, size=40_000, rng=rng)
mean_hat, se_hat = monte_carlo_mean(bernoulli_draws)
print(f"estimate={mean_hat:.4f}, stderr={se_hat:.4f}")

# ---- [cell 3] ----------------------------------------
def sample_gaussian(mean, cov, size, *, rng):
    return rng.multivariate_normal(mean=mean, cov=cov, size=size).astype(np.float64)

corr = sample_gaussian([0.0, 0.0], [[1.0, 0.8], [0.8, 1.0]], 20_000, rng=rng)
indep = sample_gaussian([0.0, 0.0], np.eye(2), 20_000, rng=rng)
print('cov (corr):', np.cov(corr, rowvar=False))
print('cov (indep):', np.cov(indep, rowvar=False))

# ---- [cell 4] ----------------------------------------
theta = rng.beta(2.0, 5.0, size=100_000)
flips = rng.binomial(1, theta).astype(np.float64)
overall = flips.mean()
conditional = theta.mean()
print(f"overall mean={overall:.4f}, conditional mean={conditional:.4f}")

# ---- [cell 5] ----------------------------------------
def running_mean(samples):
    cumsum = np.cumsum(samples, dtype=np.float64)
    return cumsum / np.arange(1, len(samples) + 1, dtype=np.float64)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for seed in range(5):
    local = np.random.default_rng(seed)
    draws = local.binomial(1, 0.3, size=5_000)
    axes[0].plot(running_mean(draws), lw=1.0, alpha=0.9, label=f'seed {seed}')
axes[0].axhline(0.3, color='#2a4d69', linestyle='--', label='true mean')
axes[0].set_title('Running mean (LLN)')
axes[0].legend(loc='best', fontsize=9)

sample_sizes = np.arange(200, 5_200, 200)
normalized = []
for _ in range(300):
    draws = rng.binomial(1, 0.3, size=sample_sizes[-1]).astype(np.float64)
    cum = np.cumsum(draws, dtype=np.float64)
    means = cum[sample_sizes - 1] / sample_sizes
    z = (means - 0.3) * np.sqrt(sample_sizes / (0.3 * 0.7))
    normalized.extend(z)
normalized = np.asarray(normalized, dtype=np.float64)
axes[1].hist(normalized, bins=50, density=True, alpha=0.6, color='#1b998b')
x = np.linspace(-4, 4, 400)
axes[1].plot(x, np.exp(-0.5 * x**2) / np.sqrt(2*np.pi), color='#e84855', lw=2.0)
axes[1].set_title('Normalised errors (CLT)')
plt.tight_layout()
plt.show()
