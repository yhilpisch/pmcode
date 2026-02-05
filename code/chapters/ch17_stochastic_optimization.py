# Python & Mathematics for Data Science and Machine Learning
# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# AI-powered by GPT-5.x

# ---- [cell 1] ----------------------------------------
# [magics stripped] %config InlineBackend.figure_format = 'retina'
import numpy as np  # arrays, RNG
import matplotlib.pyplot as plt  # plotting
plt.style.use('seaborn-v0_8')  # house style
rng = np.random.default_rng(17)  # reproducibility seed

# ---- [cell 2] ----------------------------------------
def make_linsep(n=400, d=2):  # synthetic data generator
    X = rng.normal(size=(n, d)).astype(float)  # features ~ N(0, I)
    w_true = np.array([2.0, -1.0])  # ground truth
    logits = X @ w_true + 0.25 * rng.normal(size=n)  # noisy scores
    y = (logits > 0).astype(int)  # labels {0,1}
    return X, y  # dataset

def sigmoid(z):  # logistic link
    return 1.0 / (1.0 + np.exp(-z))  # logistic link

def loss_batch(w, Xb, yb):  # mean log-loss
    z = Xb @ w  # scores
    return float(np.mean(np.log1p(np.exp(z)) - yb * z))  # stable loss

def grad_batch(w, Xb, yb):  # gradient
    z = Xb @ w  # scores
    p = sigmoid(z)  # probabilities
    return (Xb.T @ (p - yb)) / Xb.shape[0]  # mean gradient

X, y = make_linsep()  # build dataset
print(X.shape, y.shape, y.mean())  # shapes and class balance

# ---- [cell 3] ----------------------------------------
def train(w0, X, y, epochs=40, lr=0.1, batch=None, momentum=0.0):  # trainer
    w = w0.copy().astype(float)  # working params
    n = X.shape[0]  # samples
    v = np.zeros_like(w)  # momentum buffer
    his = []  # (epoch, loss)
    for e in range(epochs):  # epochs
        if batch is None:  # full batch
            g = grad_batch(w, X, y)  # full gradient
            w = w - lr * g  # update
        else:  # SGD
            idx = rng.permutation(n)  # shuffle
            for i in range(0, n, batch):  # batches
                b = idx[i:i+batch]  # batch slice
                g = grad_batch(w, X[b], y[b])  # batch grad
                v = momentum * v + g  # momentum
                w = w - lr * v  # step
        his.append((e, loss_batch(w, X, y)))  # track loss
    return w, np.array(his)  # (params, history)

# ---- [cell 4] ----------------------------------------
w0 = np.zeros(X.shape[1])  # init
_, H_full = train(w0, X, y, epochs=40, lr=0.5, batch=None)  # full GD
_, H_sgd  = train(w0, X, y, epochs=40, lr=0.1, batch=32)  # SGD
_, H_mom  = train(w0, X, y, epochs=40, lr=0.1, batch=32, momentum=0.9)  # SGD+mom
print('final_losses:', H_full[-1,1], H_sgd[-1,1], H_mom[-1,1])  # summary

# ---- [cell 5] ----------------------------------------
def grad_stats(w, X, y, B, trials=200):  # E[g], Var[g]
    n = X.shape[0]  # samples
    G = []  # collect gradients
    for _ in range(trials):  # trials
        b = rng.choice(n, size=B, replace=False)  # batch idx
        G.append(grad_batch(w, X[b], y[b]))  # grad sample
    G = np.stack(G)  # (trials, d)
    return G.mean(0), G.var(0)  # mean, variance

w_ref = np.zeros(X.shape[1])  # probe near origin
g_full = grad_batch(w_ref, X, y)  # full gradient
for B in (8, 16, 32, 64, 128):  # batch sizes
    g_mean, g_var = grad_stats(w_ref, X, y, B)  # stats
    unbiased = np.allclose(g_mean, g_full, atol=5e-3)
    print(f"B={B:>3} unbiased~{unbiased} var_sum={g_var.sum():.3e}")  # summary
