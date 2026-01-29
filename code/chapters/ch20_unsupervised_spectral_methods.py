# Python & Mathematics for Data Science and Machine Learning
# (c) Dr. Yves J. Hilpisch | The Python Quants GmbH
# AI-powered by GPT-5.x

# ---- [cell 1] ----------------------------------------
# [magics stripped] %config InlineBackend.figure_format = 'retina'
import numpy as np, matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8'); rs=np.random.default_rng(20)

# ---- [cell 2] ----------------------------------------
def kmeans(X,k,iters=50,seed=0):
    n,_=X.shape
    r=np.random.default_rng(seed)
    C=X[r.choice(n,size=k,replace=False)].copy()
    for _ in range(iters):
        d2=((X[:,None,:]-C[None,:,:])**2).sum(axis=2); a=np.argmin(d2,axis=1)
        Cnew=np.stack([X[a==j].mean(0) if (a==j).any() else C[j] for j in range(k)])
        if np.allclose(Cnew,C): break; C=Cnew
    return C,a
def wcss(X,C,a): return float(((X-C[a])**2).sum())
X = rs.normal(size=(600,2)) @ np.array([[2.0,0.6],[0.0,0.6]])
ks=range(1,8); vals=[]
for k in ks: C,a=kmeans(X,k); vals.append(wcss(X,C,a))
plt.figure(figsize=(5,4))
plt.plot(list(ks),vals,'o-')
plt.xlabel('k')
plt.ylabel('WCSS')
plt.title('Elbow')
plt.show()

# ---- [cell 3] ----------------------------------------
X2 = rs.normal(size=(600,2)) @ np.array([[2.2,0.0],[0.8,0.4]])
Xc=X2-X2.mean(0,keepdims=True)
U,S,Vt=np.linalg.svd(Xc,full_matrices=False)
scores=U[:,:1]*S[:1]
plt.figure(figsize=(5.8,4))
plt.scatter(X2[:,0],X2[:,1],s=10)
mu=X2.mean(0)
v=Vt[0]
plt.plot([mu[0]-3*v[0],mu[0]+3*v[0]],[mu[1]-3*v[1],mu[1]+3*v[1]],lw=2)
plt.title('Principal axis')
plt.show()
plt.figure(figsize=(5.8,4))
plt.hist(scores[:,0],bins=40)
plt.title('Projection on PC1')
plt.show()

# ---- [cell 4] ----------------------------------------
def two_moons(n=600,noise=0.06):
    n2=n//2; t1=np.linspace(0,np.pi,n2); t2=np.linspace(0,np.pi,n-n2)
    m1=np.c_[np.cos(t1), np.sin(t1)]; m2=np.c_[1-np.cos(t2), -np.sin(t2)-0.5]
    X=np.vstack([m1,m2]) + noise*rs.normal(size=(n,2))
    y=np.r_[np.zeros(n2,int), np.ones(n-n2,int)]
    return X,y
def spectral_embed(X,gamma=1.0,k=2):
    n=len(X)
    d2=((X[:,None,:]-X[None,:,:])**2).sum(axis=2)
    W=np.exp(-d2/gamma)
    np.fill_diagonal(W,0.0)
    D=np.diag(W.sum(1))
    Dmh=np.diag(1.0/(np.sqrt(np.diag(D))+1e-12))
    L=np.eye(n)-Dmh@W@Dmh
    S,U=np.linalg.eigh(L)
    Z=U[:,:k]
    Z=Z/(np.linalg.norm(Z,axis=1,keepdims=True)+1e-12)
    return Z
Xm, ym = two_moons(); _,a_km = kmeans(Xm,2); Z=spectral_embed(Xm); _,a_sp=kmeans(Z,2)
fig,ax=plt.subplots(1,2,figsize=(10,4))
ax[0].scatter(Xm[:,0],Xm[:,1],c=a_km,s=12,cmap='tab10')
ax[0].set_title('k-means')
ax[1].scatter(Xm[:,0],Xm[:,1],c=a_sp,s=12,cmap='tab10')
ax[1].set_title('spectral + k-means')
plt.show()
