# Python & Mathematics for Data Science and Machine Learning
# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# AI-powered by GPT-5.x

# ---- [cell 1] ----------------------------------------
# [magics stripped] %config InlineBackend.figure_format = 'retina'
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.style.use('seaborn-v0_8')
rng = np.random.default_rng(1)

# ---- [cell 2] ----------------------------------------
dist_specs = {
    'normal': stats.norm(loc=0.0, scale=1.0),
    'poisson': stats.poisson(mu=3.0),
    'exponential': stats.expon(scale=0.5),
    'beta': stats.beta(a=2.0, b=5.0),
}

for name, dist in dist_specs.items():
    samples = dist.rvs(size=40_000, random_state=rng).astype(np.float64)
    mean_emp = samples.mean()
    var_emp = samples.var(ddof=1)
    msg = (
        f"{name:10s} mean: {mean_emp:.3f} vs {dist.mean():.3f}  "
        f"var: {var_emp:.3f} vs {dist.var():.3f}"
    )
    print(msg)

# ---- [cell 3] ----------------------------------------
def logsumexp_naive(arr):
    return np.log(np.sum(np.exp(arr)))

def logsumexp_stable(arr):
    m = np.max(arr)
    return m + np.log(np.sum(np.exp(arr - m)))

log_probs = np.array([-120.0, -122.0, -119.5], dtype=np.float64)
print('naive:', logsumexp_naive(log_probs))
print('stable:', logsumexp_stable(log_probs))

# ---- [cell 4] ----------------------------------------
gamma_dist = stats.gamma(a=3.0, scale=2.0)
uniform_draws = rng.random(100_000)
gamma_samples = gamma_dist.ppf(uniform_draws)
ks_stat, ks_p = stats.kstest(gamma_samples, gamma_dist.cdf)
print(
    f"KS p-value={ks_p:.3f}  empirical mean={gamma_samples.mean():.3f}  "
    f"theory={gamma_dist.mean():.3f}"
)

# ---- [cell 5] ----------------------------------------
x = np.linspace(-4, 4, 400)
plt.figure(figsize=(10, 4))
plt.plot(x, stats.norm().pdf(x), label='Normal(0,1)', lw=2.0)
plt.plot(
    x,
    stats.gamma(a=2.0, scale=1.0).pdf(x.clip(min=0)),
    label='Gamma(2,1)',
    lw=2.0,
)
plt.xlim(-4, 8)
plt.legend(loc='upper right')
plt.title('Continuous PDFs')
plt.show()

k = np.arange(0, 12)
plt.figure(figsize=(6, 4))
plt.bar(k - 0.15, stats.poisson(mu=3).pmf(k), width=0.3, label='Poisson(3)')
plt.bar(k + 0.15, stats.poisson(mu=6).pmf(k), width=0.3, label='Poisson(6)')
plt.title('Poisson PMFs')
plt.xlabel('k')
plt.legend(loc='upper right')
plt.show()
