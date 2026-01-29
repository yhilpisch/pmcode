# Python & Mathematics for Data Science and Machine Learning
# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# AI-powered by GPT-5.x

# ---- [cell 1] ----------------------------------------
import numpy as np
np.set_printoptions(precision=4, suppress=True)

# ---- [cell 2] ----------------------------------------
# PCA denoising: MSE vs rank k
rs = np.random.default_rng(24)
h = w = 16; n=120
# bars basis
B=[]
for y in (3,7,11):
    img=np.zeros((h,w)); img[y,:]=1.0; B.append(img)
for x in (4,8,12):
    img=np.zeros((h,w)); img[:,x]=1.0; B.append(img)
B=np.stack(B,0); m=B.shape[0]
coeffs=(rs.uniform(size=(n,m))>0.6).astype(float)*rs.uniform(0.5,1.0,size=(n,m))
Xc=coeffs @ B.reshape(m,-1); Xc/=Xc.max()+1e-12; Xn=Xc+0.15*rs.normal(size=Xc.shape)
ks=[1,2,4,8,12]
errs=[]
X0=Xn - Xn.mean(0,keepdims=True)
U,S,Vt=np.linalg.svd(X0, full_matrices=False)
for k in ks:
    Xk=(U[:,:k]*S[:k]) @ Vt[:k] + Xn.mean(0,keepdims=True)
    errs.append(float(np.mean((Xc - Xk)**2)))
print('MSE vs k:', {k:round(e,4) for k,e in zip(ks,errs)})

# ---- [cell 3] ----------------------------------------
# Calibration: ECE raw vs temperature scaled
rs = np.random.default_rng(24)
N=2000; d=2; pr=0.12; n_pos=int(N*pr)
Xp=rs.normal([1.2,1.0],1.0,size=(n_pos,d))
Xn=rs.normal([0.0,0.0],1.0,size=(N-n_pos,d))
X=np.vstack([Xp,Xn]); y=np.r_[np.ones(n_pos,int), np.zeros(N-n_pos,int)]
idx=rs.permutation(N); X=X[idx]; y=y[idx]

def sigmoid(z): return 1/(1+np.exp(-z))

w=np.zeros(d); b=0.0
for _ in range(400):
    z=X@w+b; p=sigmoid(z)
    w -= 0.2 * ((X.T @ (p - y))/N + 1e-3*w)
    b -= 0.2 * float(np.mean(p - y))

edges=np.linspace(0,1,11)
ind=np.clip(np.searchsorted(edges, sigmoid(X@w+b), 'right')-1, 0, 9)
def ECE(q):
    total = 0.0
    for k in range(10):
        if np.any(ind == k):
            acc = float(np.mean(y[ind == k]))
            conf = float(np.mean(q[ind == k]))
            total += (np.sum(ind == k) / len(q)) * abs(acc - conf)
    return total
p=sigmoid(X@w+b); pT=sigmoid((X@w+b)/1.5)
print('ECE raw, temp =', round(ECE(p),3), round(ECE(pT),3))

# ---- [cell 4] ----------------------------------------
# Seed sensitivity: mean±sd of log-loss across seeds
rs = np.random.default_rng(24)
vals=[]
for s in range(10,16):
    r=np.random.default_rng(s)
    Xp=r.normal([1.2,1.0],1.0,size=(n_pos,d))
    Xn=r.normal([0.0,0.0],1.0,size=(N-n_pos,d))
    X=np.vstack([Xp,Xn]); y=np.r_[np.ones(n_pos,int), np.zeros(N-n_pos,int)]
    idx=r.permutation(N); X=X[idx]; y=y[idx]
    w=np.zeros(d); b=0.0
    for _ in range(300):
        z=X@w+b; p=sigmoid(z)
        w -= 0.2 * ((X.T @ (p - y))/N + 1e-3*w)
        b -= 0.2 * float(np.mean(p - y))
    def logloss(y, q):
        q = np.clip(q, 1e-12, 1 - 1e-12)
        return float(np.mean(-(y * np.log(q) + (1 - y) * np.log(1 - q))))
    vals.append(logloss(y[int(0.7*N):], sigmoid((X[int(0.7*N):]@w+b))))
print(
    "mean, sd =",
    round(float(np.mean(vals)), 3),
    round(float(np.std(vals, ddof=1)), 3),
)
