# Python & Mathematics for Data Science and Machine Learning
# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# AI-powered by GPT-5.x

# ---- [cell 1] ----------------------------------------
import numpy as np
rs = np.random.default_rng(26)
X = np.linspace(-1, 1, 64)[:, None]
y = np.sin(3*X) + 0.1*rs.normal(size=X.shape)
# 1-hidden layer tanh network: 1→16→1
W1 = rs.normal(size=(1,16)); b1 = np.zeros((16,))
W2 = rs.normal(size=(16,1)); b2 = np.zeros((1,))

def tanh(x):
    return np.tanh(x)

for _ in range(200):
    H = tanh(X@W1 + b1)
    yhat = H@W2 + b2
    err = yhat - y
    dW2 = H.T @ err / len(X); db2 = err.mean(axis=0)
    dH = err @ W2.T
    dZ = dH * (1 - H**2)
    dW1 = X.T @ dZ / len(X); db1 = dZ.mean(axis=0)
    W2 -= 1e-2 * dW2; b2 -= 1e-2 * db2
    W1 -= 1e-2 * dW1; b1 -= 1e-2 * db1
loss = float((err**2).mean())
print('final MSE =', round(loss, 4))
