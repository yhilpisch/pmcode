# Python & Mathematics for Data Science and Machine Learning
# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# AI-powered by GPT-5

"""Generated from notebook: /Users/yves/Library/CloudStorage/Dropbox/Program/books/4_pm/notebooks/ch19_classification_calibration.ipynb

Do not edit by hand — re-generate via tools/export_chapters_from_notebooks.py.
"""

# ---- [cell 1] ----------------------------------------
# [magics stripped] %config InlineBackend.figure_format = 'retina'
import numpy as np, matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8'); rng=np.random.default_rng(19)

# ---- [cell 2] ----------------------------------------
def make(n=4000,pos_rate=0.15):
    n_pos=int(n*pos_rate); n_neg=n-n_pos
    Xp=rng.normal([1.2,1.0],1.0,(n_pos,2)); Xn=rng.normal([0.0,0.0],1.0,(n_neg,2))
    X=np.vstack([Xp,Xn]); y=np.r_[np.ones(n_pos,int), np.zeros(n_neg,int)]
    idx=rng.permutation(n); return X[idx], y[idx]
def sigmoid(z): return 1.0/(1.0+np.exp(-z))
def fit_logreg(X,y,lr=0.2,it=500,lam=1e-3):
    n,d=X.shape; w=np.zeros(d); b=0.0
    for _ in range(it):
        z=X@w+b; p=sigmoid(z); w -= lr*((X.T@(p-y))/n + lam*w); b -= lr*float((p-y).sum()/n)
    return w,b
X,y = make(); n=len(X); tr=np.arange(int(0.7*n)); te=np.arange(int(0.7*n),n)
w,b = fit_logreg(X[tr], y[tr]); z=X[te]@w+b; p=sigmoid(z)
plt.figure(figsize=(5,4)); plt.scatter(X[te,0],X[te,1],c=p,s=10,cmap='viridis'); plt.colorbar(label='p_hat'); plt.title('Scores (probabilities)'); plt.show()

# ---- [cell 3] ----------------------------------------
def roc_pr(y, s):
    o=np.argsort(s)[::-1]; y=y[o]; s=s[o]; P=(y==1).sum(); N=(y==0).sum()
    tp=fp=0; ROC=[]; PR=[]
    prev=None
    for i in range(len(s)):
        if prev is None or s[i]!=prev:
            if i>0: ROC.append((fp/N if N else 0, tp/P if P else 0)); PR.append((tp/P if P else 0, tp/(tp+fp) if tp+fp else 1.0))
            prev=s[i]
        if y[i]==1: tp+=1
        else: fp+=1
    ROC.append((fp/N if N else 0, tp/P if P else 0)); PR.append((tp/P if P else 0, tp/(tp+fp) if tp+fp else 1.0))
    return np.array(ROC), np.array(PR)
ro, pr = roc_pr(y[te], p)
fig,ax=plt.subplots(1,2,figsize=(10,4)); ax[0].plot(ro[:,0], ro[:,1]); ax[0].plot([0,1],[0,1],'--',lw=1); ax[0].set_title('ROC'); ax[0].set_xlabel('FPR'); ax[0].set_ylabel('TPR');
ax[1].plot(pr[:,0], pr[:,1]); ax[1].set_title('PR'); ax[1].set_xlabel('Recall'); ax[1].set_ylabel('Precision'); plt.show()

# ---- [cell 4] ----------------------------------------
def ece(y,p,bins=10):
    edges=np.linspace(0,1,bins+1); inds=np.clip(np.searchsorted(edges,p,'right')-1,0,bins-1)
    e=0.0
    for k in range(bins):
        m=inds==k; n_k=int(m.sum());
        if n_k==0: continue
        acc=float(y[m].mean()); conf=float(p[m].mean()); e += (n_k/len(p))*abs(acc-conf)
    return float(e)
T=1.6; pT=sigmoid(z/T); print('ECE raw', round(ece(y[te],p),3), 'ECE temp', round(ece(y[te],pT),3))
# reliability diagram
edges=np.linspace(0,1,11); inds=np.clip(np.searchsorted(edges,pT,'right')-1,0,9)
conf=[pT[inds==k].mean() if (inds==k).any() else np.nan for k in range(10)]
acc=[y[te][inds==k].mean() if (inds==k).any() else np.nan for k in range(10)]
plt.figure(figsize=(5.8,4)); plt.plot([0,1],[0,1],'--',lw=1,color='k'); plt.plot(conf,acc,'o-'); plt.xlabel('mean confidence'); plt.ylabel('empirical accuracy'); plt.title('Reliability (temp-scaled)'); plt.show()
