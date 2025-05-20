#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import r2_score, mean_squared_error


def generate_map(W=10, H=10, p_obstacle=0.2, conn_thresh=0.8):
    while True:
        grid = np.zeros((H, W), dtype=int)
        grid[0,:] = grid[-1,:] = grid[:,0] = grid[:,-1] = 1
        for i in range(1, H-1):
            for j in range(1, W-1):
                if np.random.rand() < p_obstacle:
                    grid[i,j] = 1
        opens = [(i,j) for i in range(H) for j in range(W) if grid[i,j]==0]
        if not opens: continue
        vis, dq = {opens[0]}, deque([opens[0]])
        while dq:
            ci,cj = dq.popleft()
            for di,dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                nb = (ci+di, cj+dj)
                if nb in opens and nb not in vis:
                    vis.add(nb); dq.append(nb)
        if len(vis)/len(opens) >= conn_thresh:
            return grid

def update_L(grid, L, action):
    H,W = grid.shape
    di,dj = {'up':(-1,0),'down':(1,0),'left':(0,-1),'right':(0,1)}[action]
    new = set()
    for (i,j) in L:
        ni,nj = i+di, j+dj
        if 0<=ni<H and 0<=nj<W and grid[ni,nj]==0:
            new.add((ni,nj))
        else:
            new.add((i,j))
    return new

def simulate_pi0(grid, L0, max_steps=500):
    L = set(L0)
    true = random.choice(list(L))
    moves = 0
    while len(L)>1 and moves<max_steps:
        guess = random.choice(list(L))
        dq, seen = deque([(true, [])]), {true}
        path = None
        while dq and path is None:
            (ci, cj), p = dq.popleft()
            if (ci, cj) == guess:
                path = p; break
            for mv, (di, dj) in [
                ('up', (-1, 0)), ('down', (1, 0)),
                ('left', (0, -1)), ('right', (0, 1))
            ]:
                ni, nj = ci+di, cj+dj
                if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1] \
                   and grid[ni, nj] == 0 and (ni, nj) not in seen:
                    seen.add((ni, nj))
                    dq.append(((ni, nj), p + [mv]))
        if path is None:
            path = [random.choice(['up','down','left','right'])]
        for mv in path:
            di,dj = {'up':(-1,0),'down':(1,0),'left':(0,-1),'right':(0,1)}[mv]
            nb = (true[0]+di, true[1]+dj)
            if grid[nb] == 0:
                true = nb
            L = update_L(grid, L, mv)
            moves += 1
            if len(L) == 1 or moves >= max_steps:
                break
    return moves

def compute_move_viability(grid):
    H,W = grid.shape
    mv = np.zeros((4,H,W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            if grid[i,j]==0:
                if i>0   and grid[i-1,j]==0: mv[0,i,j]=1.0
                if i<H-1 and grid[i+1,j]==0: mv[1,i,j]=1.0
                if j>0   and grid[i,j-1]==0: mv[2,i,j]=1.0
                if j<W-1 and grid[i,j+1]==0: mv[3,i,j]=1.0
    return mv


class BeliefDatasetEnhanced(Dataset):
    def __init__(self, N=10000, grid_size=(10,10), seed=0):
        random.seed(seed); np.random.seed(seed)
        self.data = []
        H,W = grid_size
        for _ in range(N):
            grid = generate_map(W,H)
            opens = [(i,j) for i in range(H) for j in range(W) if grid[i,j]==0]
            L0 = set(random.sample(opens, random.choice([2,4,8,16,32])))
            c  = simulate_pi0(grid, L0)
            bm = np.zeros_like(grid, dtype=np.float32)
            for (i,j) in L0: bm[i,j]=1.0
            mv = compute_move_viability(grid)
            size_L = max(len(L0),1)
            fp = bm / size_L
            ent= np.full_like(grid, np.log(size_L), dtype=np.float32)
            X  = np.stack([grid.astype(np.float32), bm, *mv, fp, ent], axis=0)
            for ch in range(X.shape[0]):
                m,M = X[ch].min(), X[ch].max()
                if M>m: X[ch] = (X[ch]-m)/(M-m)-0.5
            sz = np.array([len(L0)/(H*W)], dtype=np.float32)
            self.data.append((X, sz, np.array([c],dtype=np.float32)))

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        X,sz,c = self.data[idx]
        return torch.tensor(X), torch.tensor(sz), torch.tensor(c)


class CostNetEnhanced(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(8, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32,64,3,padding=1),    nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64+1, 1)
    def forward(self,x,sz):
        h = self.conv(x).view(x.size(0),-1)
        h = torch.cat([h, sz], dim=1)
        return self.fc(h)

def make_input_tensor(grid, Lset):
    bm = np.zeros_like(grid, dtype=np.float32)
    for (i,j) in Lset: bm[i,j]=1.0
    mv = compute_move_viability(grid)
    size_L = max(len(Lset),1)
    fp = bm / size_L
    ent= np.full_like(grid, np.log(size_L), dtype=np.float32)
    X  = np.stack([grid.astype(np.float32), bm, *mv, fp, ent], axis=0)
    for ch in range(X.shape[0]):
        m,M = X[ch].min(), X[ch].max()
        if M>m: X[ch] = (X[ch]-m)/(M-m)-0.5
    sz = np.array([len(Lset)/(grid.size)], dtype=np.float32)
    return torch.tensor(X[None]), torch.tensor(sz[None])


if __name__=="__main__":
    device = torch.device("cpu")

    ds = BeliefDatasetEnhanced()
    n = len(ds)
    n_train = int(0.7*n); n_val=int(0.15*n); n_test=n-n_train-n_val
    train_ds, val_ds, test_ds = random_split(ds, [n_train,n_val,n_test])
    train_ld = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_ld   = DataLoader(val_ds,   batch_size=64)
    test_ld  = DataLoader(test_ds,  batch_size=64)

    model = CostNetEnhanced().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    crit = nn.MSELoss()

    best_val, wait = float('inf'), 0
    train_losses, val_losses = [], []
    for epoch in range(1, 31):
        model.train(); tr=0
        for X,sz,y in train_ld:
            X,sz,y = X.to(device), sz.to(device), y.to(device)
            pred = model(X,sz)
            loss = crit(pred,y)
            opt.zero_grad(); loss.backward(); opt.step()
            tr+= loss.item()*X.size(0)
        train_losses.append(tr/len(train_ds))

        model.eval(); vl=0
        with torch.no_grad():
            for X,sz,y in val_ld:
                vl += crit(model(X.to(device), sz.to(device)), y.to(device)).item()*X.size(0)
        val_losses.append(vl/len(val_ds))

        if val_losses[-1]<best_val:
            best_val, wait = val_losses[-1], 0
            torch.save(model.state_dict(),"costnet_pre.pt")
        else:
            wait+=1
            if wait>=5: break

    model.load_state_dict(torch.load("costnet_pre.pt", map_location=device))
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for X,sz,y in test_ld:
            out = model(X.to(device), sz.to(device)).cpu().numpy().flatten()
            preds.extend(out); ys.extend(y.numpy().flatten())
    print("Pre-train R²:", r2_score(ys, preds))

    for n in [2,4,8,16,32]:
        m0, m1 = [], []
        for _ in range(50):
            g = generate_map()
            opens = [(i,j) for i in range(10) for j in range(10) if g[i,j]==0]
            L0 = set(random.sample(opens,n))
            m0.append(simulate_pi0(g, L0))
            best,bval=None,float('inf')
            for a in ['up','down','left','right']:
                La = update_L(g, L0, a)
                X1, sz1 = make_input_tensor(g, La)
                c = model(X1.to(device), sz1.to(device)).item()
                if c<bval: bval,best = c,a
            if best is None: best=random.choice(['up','down','left','right'])
            L1 = update_L(g, L0, best)
            m1.append(1 + simulate_pi0(g, L1))
        print(f"|L|={n}: π₀={np.mean(m0):.1f}, π₁={np.mean(m1):.1f}")

    opt = optim.Adam(model.parameters(), lr=1e-4)
    bellman_losses = []
    for epoch in range(1,21):
        model.train(); tot=0
        for X,sz,_ in train_ld:
            X,sz = X.to(device), sz.to(device)
            with torch.no_grad():
                succ = []
                for a in ['up','down','left','right']:
                    shifted = []
                    for x_tensor, sz_val in zip(X.cpu(), sz.cpu()):
                        grid = (x_tensor[0]>0).numpy().astype(int)
                        belief = {(i,j) for i in range(10) for j in range(10) if x_tensor[1,i,j]>0.0}
                        La = update_L(grid, belief, a)
                        Xn, sz_n = make_input_tensor(grid, La)
                        shifted.append((Xn[0], sz_n[0]))
                    Xs = torch.stack([x for x,_ in shifted]).to(device)
                    Sz = torch.stack([s for _,s in shifted]).to(device)
                    succ.append(model(Xs, Sz))
                target = 1 + torch.min(torch.stack(succ), dim=0)[0]
            pred = model(X, sz)
            loss = nn.MSELoss()(pred, target)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()*X.size(0)
        bellman_losses.append(tot/len(train_ds))
        print(f"Bellman Epoch {epoch}, MSE={bellman_losses[-1]:.2f}")

    for n in [2,4,8,16,32]:
        m0,m1,mstar = [],[],[]
        for _ in range(50):
            g = generate_map()
            opens = [(i,j) for i in range(10) for j in range(10) if g[i,j]==0]
            L0 = set(random.sample(opens,n))
            m0.append(simulate_pi0(g,L0))
            best,bval=None,float('inf')
            for a in ['up','down','left','right']:
                La = update_L(g, L0, a)
                X1, sz1 = make_input_tensor(g, La)
                c = model(X1.to(device), sz1.to(device)).item()
                if c<bval: bval,best = c,a
            if best is None: best=random.choice(['up','down','left','right'])
            L1 = update_L(g, L0, best); m1.append(1+simulate_pi0(g, L1))
            L=L0; moves=0
            while len(L)>1 and moves<500:
                bmv,bval=None,float('inf')
                for a in ['up','down','left','right']:
                    La=update_L(g,L,a)
                    Xp, szp = make_input_tensor(g, La)
                    cp = model(Xp.to(device), szp.to(device)).item()
                    if cp<bval: bval,bmv = cp,a
                L = update_L(g,L,bmv); moves+=1
            mstar.append(moves)
        print(f"|L|={n}: π₀={np.mean(m0):.1f}, π₁={np.mean(m1):.1f}, π*={np.mean(mstar):.1f}")


# In[2]:


import matplotlib.pyplot as plt
import numpy as np

L_sizes = [2, 4, 8, 16, 32]
avg_moves_pi0 = [18.6, 55.7, 97.6, 123.3, 177.6]

plt.figure(figsize=(6,4))
plt.plot(L_sizes, avg_moves_pi0, marker='o', linestyle='-')
plt.xlabel('|L| (initial belief size)')
plt.ylabel('Average moves to localize')
plt.title('π₀ Performance: Avg Moves vs |L|')
plt.grid(True)
plt.savefig('fig1_pi0_performance_updated.png', dpi=300)
plt.show()

train_losses = [tr for tr in train_losses]  
val_losses   = [vl for vl in val_losses]

plt.figure(figsize=(6,4))
plt.plot(train_losses, label='Train MSE')
plt.plot(val_losses,   label='Val   MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('Supervised Pre-training Loss')
plt.legend()
plt.grid(True)
plt.savefig('fig2a_pretrain_loss_updated.png', dpi=300)
plt.show()


ys = np.array(ys)     
preds = np.array(preds)  

plt.figure(figsize=(6,6))
plt.scatter(ys, preds, s=10, alpha=0.3)
m = max(ys.max(), preds.max())
plt.plot([0, m], [0, m], 'r--')
plt.xlabel('True rollout cost')
plt.ylabel('Predicted cost')
plt.title('Pre-training: True vs Predicted')
plt.grid(True)
plt.savefig('fig2b_true_vs_pred_updated.png', dpi=300)
plt.show()


bellman_losses = bellman_losses  

plt.figure(figsize=(6,4))
plt.plot(bellman_losses, marker='o')
plt.xlabel('Bellman fine-tune epoch')
plt.ylabel('MSE of C(L) - [1 + min_a C(L_a)]')
plt.title('Bellman Residual Loss')
plt.grid(True)
plt.savefig('fig3_bellman_loss_updated.png', dpi=300)
plt.show()

policy_results = {
    'L_size': [2, 4, 8, 16, 32],
    'pi0':     [28.6, 86.4, 64.1, 109.6, 138.1],
    'pi1':     [29.6, 86.8, 63.9, 114.9, 136.3],
    'pistar': [500.0, 500.0, 500.0, 500.0, 500.0]
}

sizes = np.array(policy_results['L_size'])
pi0   = np.array(policy_results['pi0'])
pi1   = np.array(policy_results['pi1'])
pistar= np.array(policy_results['pistar'])
x = np.arange(len(sizes))
width = 0.25

plt.figure(figsize=(6,4))
plt.bar(x - width, pi0,    width, label='π₀')
plt.bar(x,        pi1,    width, label='π₁')
plt.bar(x + width, pistar,width, label='π*')
plt.xticks(x, sizes)
plt.xlabel('|L| (initial belief size)')
plt.ylabel('Average moves to localize')
plt.title('Policy Comparison: π₀ vs π₁ vs π*')
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('fig4_policy_comparison_updated.png', dpi=300)
plt.show()


# In[ ]:




