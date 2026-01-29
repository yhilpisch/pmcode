# Python & Mathematics for Data Science and Machine Learning
# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# AI-powered by GPT-5.x

# ---- [cell 1] ----------------------------------------
import sys, json, platform, hashlib
import numpy as np
np.set_printoptions(precision=6, suppress=True)
SEED=25; rs=np.random.default_rng(SEED)

# data + hash
N=2000; d=2; pr=0.2; n_pos=int(N*pr)
Xp=rs.normal([1.0,1.0],1.0,size=(n_pos,d))
Xn=rs.normal([0.0,0.0],1.0,size=(N-n_pos,d))
X=np.vstack([Xp,Xn]); y=np.r_[np.ones(n_pos,int), np.zeros(N-n_pos,int)]
tr=np.arange(int(0.7*N)); te=np.arange(int(0.7*N),N)
h=hashlib.sha256()
h.update(X.astype(np.float64).tobytes(order='C'))
h.update(y.astype(np.int64).tobytes(order='C'))
DATA=h.hexdigest()[:16]
CODE='draft-ch25'
ENV = (
    f"py{sys.version_info.major}.{sys.version_info.minor}-"
    f"np{np.__version__}-os{platform.system()}"
)
print(json.dumps({'seed':SEED,'data':DATA,'code':CODE,'env':ENV}, sort_keys=True))

# ---- [cell 2] ----------------------------------------
def sigmoid(z): return 1/(1+np.exp(-z))
w=np.zeros(d); b=0.0
for _ in range(500):
    z=X[tr]@w+b; p=sigmoid(z)
    w -= 0.2 * ((X[tr].T @ (p - y[tr]))/len(tr) + 1e-3*w)
    b -= 0.2 * float(np.mean(p - y[tr]))
q=sigmoid(X[te]@w+b)
q=np.clip(q,1e-12,1-1e-12)
val=float(np.mean(-(y[te]*np.log(q)+(1-y[te])*np.log(1-q))))
print('val_logloss=', round(val,4))
