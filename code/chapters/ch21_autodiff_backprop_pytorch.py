# Python & Mathematics for Data Science and Machine Learning
# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# AI-powered by GPT-5

"""Generated from notebook: /Users/yves/Library/CloudStorage/Dropbox/Program/books/4_pm/notebooks/ch21_autodiff_backprop_pytorch.ipynb

Do not edit by hand — re-generate via tools/export_chapters_from_notebooks.py.
"""

# ---- [cell 1] ----------------------------------------
# [magics stripped] %config InlineBackend.figure_format = 'retina'
import numpy as np, matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')

# ---- [cell 2] ----------------------------------------
import numpy as np  # arrays and math

class Node:
    def __init__(self, value, parents=(), backward=lambda g: None, name=None):
        self.value = np.asarray(value, dtype=float)
        self.parents = parents
        self.backward_fn = backward
        self.grad = np.zeros_like(self.value)
        self.name = name

    def backward(self, grad=None):
        topo = []
        seen = set()
        def build(v):
            if id(v) in seen:
                return
            seen.add(id(v))
            for p in v.parents:
                build(p)
            topo.append(v)
        build(self)
        self.grad[...] = 1.0 if grad is None else grad
        for v in reversed(topo):
            v.backward_fn(v.grad)

# Primitives (each returns a Node with value and backward rule)
def add(a, b):
    a = a if isinstance(a, Node) else Node(a)
    b = b if isinstance(b, Node) else Node(b)
    z = Node(a.value + b.value, parents=(a, b))
    def backward(g):
        a.grad += g
        b.grad += g
    z.backward_fn = backward
    return z

def mul(a, b):
    a = a if isinstance(a, Node) else Node(a)
    b = b if isinstance(b, Node) else Node(b)
    z = Node(a.value * b.value, parents=(a, b))
    def backward(g):
        a.grad += g * b.value
        b.grad += g * a.value
    z.backward_fn = backward
    return z

def tanh(a):
    a = a if isinstance(a, Node) else Node(a)
    t = np.tanh(a.value)
    z = Node(t, parents=(a,))
    def backward(g):
        a.grad += g * (1.0 - t * t)
    z.backward_fn = backward
    return z

def sum1(a):
    a = a if isinstance(a, Node) else Node(a)
    z = Node(np.array(a.value.sum()), parents=(a,))
    def backward(g):
        a.grad += g
    z.backward_fn = backward
    return z

# Toy network and grad check
rs = np.random.default_rng(21)
x = Node(rs.normal())
w1 = Node(rs.normal())
b1 = Node(0.0)
w2 = Node(rs.normal())
b2 = Node(0.0)
y = Node(1.0)

a1 = add(mul(w1, x), b1)
h = tanh(a1)
yhat = add(mul(w2, h), b2)
loss = sum1((yhat.value - y.value) ** 2)
loss.parents = (yhat, y)

def back(g):
    yhat.grad += g * 2.0 * (yhat.value - y.value)
    y.grad += g * -2.0 * (yhat.value - y.value)

loss.backward_fn = back
loss.backward()
print('grads', float(w1.grad), float(w2.grad))
