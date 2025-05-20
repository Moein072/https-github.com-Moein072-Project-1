#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import math
import matplotlib.pyplot as plt
from collections import deque
import numpy as np


def generate_ship(D):
   
    ship = [[0]*D for _ in range(D)]

    # (1) Random interior cell
    x = random.randint(1, D-2)
    y = random.randint(1, D-2)
    ship[x][y] = 1

    while True:
        frontier = []
        for i in range(1, D-1):
            for j in range(1, D-1):
                if ship[i][j] == 0:
                    open_neighbors = 0
                    for (dx, dy) in [(1,0), (-1,0), (0,1), (0,-1)]:
                        nx, ny = i+dx, j+dy
                        if 0 <= nx < D and 0 <= ny < D and ship[nx][ny] == 1:
                            open_neighbors += 1
                    if open_neighbors == 1:
                        frontier.append((i,j))
        if not frontier:
            break
        fx, fy = random.choice(frontier)
        ship[fx][fy] = 1

 
    dead_ends = []
    for i in range(1, D-1):
        for j in range(1, D-1):
            if ship[i][j] == 1:
                neighbor_count = 0
                for (dx, dy) in [(1,0), (-1,0), (0,1), (0,-1)]:
                    nx, ny = i+dx, j+dy
                    if 0 <= nx < D and 0 <= ny < D and ship[nx][ny] == 1:
                        neighbor_count += 1
                if neighbor_count == 1:
                    dead_ends.append((i, j))

    random.shuffle(dead_ends)
    half_count = len(dead_ends)//2
    for (dx, dy) in dead_ends[:half_count]:
        blocked_neighbors = []
        for (nx, ny) in [(dx-1,dy),(dx+1,dy),(dx,dy-1),(dx,dy+1)]:
            if 0 <= nx < D and 0 <= ny < D and ship[nx][ny] == 0:
                blocked_neighbors.append((nx, ny))
        if blocked_neighbors:
            bx, by = random.choice(blocked_neighbors)
            ship[bx][by] = 1

    return ship

def spread_fire(ship, fire_set, q):
    
    D = len(ship)
    new_fire = set()
    for (fx, fy) in fire_set:
        for (dx, dy) in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = fx+dx, fy+dy
            if 0 <= nx < D and 0 <= ny < D and ship[nx][ny] == 1:
                if (nx, ny) not in fire_set:
                    K = 0
                    for (ax, ay) in [(1,0),(-1,0),(0,1),(0,-1)]:
                        if (nx+ax, ny+ay) in fire_set:
                            K += 1
                    if K>0:
                        p_ignite = 1 - (1 - q)**K
                        if random.random()<p_ignite:
                            new_fire.add((nx, ny))
    return fire_set.union(new_fire)



def bfs_search(ship, start, goal, blocked=set()):
    if start==goal:
        return [start]
    D = len(ship)
    from collections import deque
    queue = deque([start])
    parents = {start:None}
    if start in blocked:
        return None
    while queue:
        cx, cy = queue.popleft()
        if (cx, cy) == goal:
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = parents[node]
            path.reverse()
            return path
        for (dx, dy) in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = cx+dx, cy+dy
            if 0<=nx<D and 0<=ny<D and ship[nx][ny]==1 and (nx,ny) not in blocked:
                if (nx, ny) not in parents:
                    parents[(nx, ny)] = (cx, cy)
                    queue.append((nx, ny))
    return None

def get_adj_fire(fire_set, D):
    adj = set()
    for (fx, fy) in fire_set:
        for (dx, dy) in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = fx+dx, fy+dy
            if 0<=nx<D and 0<=ny<D:
                adj.add((nx, ny))
    return adj

def choose_move(ship, bot_pos, button, fire_set, bot1plan, strat):
    if strat==1:
       
        if bot1plan:
            nxt = bot1plan[0]
            bot1plan = bot1plan[1:]
            return nxt, bot1plan
        else:
            return bot_pos, bot1plan
    elif strat==2:
       
        path2 = bfs_search(ship, bot_pos, button, blocked=fire_set)
        if path2 and len(path2)>1:
            return path2[1], bot1plan
        return bot_pos, bot1plan
    elif strat==3:
        
        D = len(ship)
        adj = get_adj_fire(fire_set, D)
        big_block = fire_set.union(adj)
        path3 = bfs_search(ship, bot_pos, button, blocked=big_block)
        if path3 is None:
            path3 = bfs_search(ship, bot_pos, button, blocked=fire_set)
        if path3 and len(path3)>1:
            return path3[1], bot1plan
        return bot_pos, bot1plan
    else:
      
        path4 = bfs_search(ship, bot_pos, button, blocked=fire_set)
        if path4 and len(path4)>1:
            return path4[1], bot1plan
        return bot_pos, bot1plan

def run_single(ship, q, strat, steps=200):
    
    D = len(ship)
    open_cells = [(r,c) for r in range(D) for c in range(D) if ship[r][c]==1]
    if len(open_cells)<3:
        return False
    random.shuffle(open_cells)
    bot_start = open_cells[0]
    fire_start= open_cells[1]
    button = open_cells[2]
    fire_set = {fire_start}

  
    bot1plan = None
    if strat==1:
        init_path = bfs_search(ship, bot_start, button, blocked=fire_set)
        if init_path:
            bot1plan = init_path[1:]
        else:
            bot1plan = []

    bot_pos = bot_start
    for t in range(steps):
        bot_pos, bot1plan = choose_move(ship, bot_pos, button, fire_set, bot1plan, strat)
        if bot_pos==button:
            return True
        fire_set = spread_fire(ship, fire_set, q)
        if bot_pos in fire_set:
            return False
    return False


def record_fire_timeline(ship, fire_start, q, steps=200):
    
    D = len(ship)
    burn_time = [[float('inf')]*D for _ in range(D)]
    x0,y0 = fire_start
    burn_time[x0][y0] = 0
    current_fire = {fire_start}
    for t in range(steps):
        new_b = set()
        for (fx,fy) in current_fire:
            for (dx,dy) in [(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = fx+dx, fy+dy
                if 0<=nx<D and 0<=ny<D and ship[nx][ny]==1:
                    if burn_time[nx][ny]==float('inf'):
                       
                        kc=0
                        for (ax,ay) in [(1,0),(-1,0),(0,1),(0,-1)]:
                            if (nx+ax, ny+ay) in current_fire:
                                kc+=1
                        if kc>0:
                            p_ignite = 1-(1-q)**kc
                            if random.random()<p_ignite:
                                burn_time[nx][ny] = t+1
                                new_b.add((nx,ny))
        if not new_b:
            break
        current_fire|=new_b
    return burn_time

def scenario_winnable(ship, burn_time, bot_start, button, max_t=200):
    
    D=len(ship)
    from collections import deque
    Q = deque()
    visited=set()

    if burn_time[bot_start[0]][bot_start[1]]==0:
        return False  
    Q.append((bot_start[0], bot_start[1],0))
    visited.add((bot_start[0], bot_start[1],0))

    while Q:
        x, y, t = Q.popleft()
        if (x,y)==button:
            return True
        if t>=max_t:
            continue
      
        for (dx,dy) in [(0,0),(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            nt = t+1
            if 0<=nx<D and 0<=ny<D and ship[nx][ny]==1:
                if nt<burn_time[nx][ny]:
                    if (nx,ny,nt) not in visited:
                        visited.add((nx,ny,nt))
                        Q.append((nx,ny,nt))
    return False

def run_experiments_with_winnability(D=15, trials=20, qvals=None):
    if qvals is None:
        qvals=[0.0,0.2,0.4,0.6,0.8,1.0]
    winnable_frac=[]
    uncond={1:[],2:[],3:[],4:[]}
    cond={1:[],2:[],3:[],4:[]}

    for q in qvals:
        scenario_count=0
        scenario_wcount=0
        bot_un={1:0,2:0,3:0,4:0}
        bot_co={1:0,2:0,3:0,4:0}
        for _ in range(trials):
            
            sp=generate_ship(D)
            
            opens=[(r,c) for r in range(D) for c in range(D) if sp[r][c]==1]
            if len(opens)<3:
               
                continue
            scenario_count+=1
            random.shuffle(opens)
            bpos=opens[0]
            fpos=opens[1]
            butpos=opens[2]

           
            burn_t=record_fire_timeline(sp,fpos,q,2*D)
            iswin=scenario_winnable(sp,burn_t,bpos,butpos,2*D)
            if iswin:
                scenario_wcount+=1

           
            for st in [1,2,3,4]:
              
                success=run_single(sp,q,st)
                if success:
                    bot_un[st]+=1
                    if iswin:
                        bot_co[st]+=1

        if scenario_count>0:
            wfrac=scenario_wcount/scenario_count
        else:
            wfrac=0
        winnable_frac.append(wfrac)
       
        for st in [1,2,3,4]:
            if scenario_count>0:
                uncd=bot_un[st]/scenario_count
            else:
                uncd=0
            uncond[st].append(uncd)
       
        for st in [1,2,3,4]:
            if scenario_wcount>0:
                conds=bot_co[st]/scenario_wcount
            else:
                conds=0
            cond[st].append(conds)

    return winnable_frac, uncond, cond, qvals

def main():
    random.seed(0)
    D=15
    trials=30
    qvals=[0.0,0.2,0.4,0.6,0.8,1.0]

    wfrac,uncond,cond,qvals=run_experiments_with_winnability(D,trials,qvals)

  
    print("Winnability vs. q:")
    for i,q in enumerate(qvals):
        print(f" q={q}, fraction={wfrac[i]*100:.1f}%")
    print()

   
    print("Unconditional success rates:")
    for st in [1,2,3,4]:
        print(f" Bot {st}:")
        for i,q in enumerate(qvals):
            print(f"   q={q}, success={uncond[st][i]*100:.1f}%")
        print()

    print("Conditional success (only winnable):")
    for st in [1,2,3,4]:
        print(f" Bot {st}:")
        for i,q in enumerate(qvals):
            print(f"   q={q}, success={cond[st][i]*100:.1f}%")
        print()

    import matplotlib.pyplot as plt

    plt.figure(figsize=(9,5))
    plt.plot(qvals, wfrac, marker='o', label='Winnable Fraction')
    plt.xlabel('q')
    plt.ylabel('Fraction')
    plt.title('Winnability vs. q')
    plt.legend()
    plt.grid(True)
    plt.show()

  
    plt.figure(figsize=(9,5))
    for st in [1,2,3,4]:
        plt.plot(qvals, uncond[st], marker='o', label=f'Bot {st}')
    plt.xlabel('q')
    plt.ylabel('Unconditional Success')
    plt.title('Unconditional Success vs. q')
    plt.legend()
    plt.grid(True)
    plt.show()

   
    plt.figure(figsize=(9,5))
    for st in [1,2,3,4]:
        plt.plot(qvals, cond[st], marker='o', label=f'Bot {st}')
    plt.xlabel('q')
    plt.ylabel('Conditional Success')
    plt.title('Conditional Success (Winnable Only) vs. q')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__=="__main__":
    main()


# In[2]:


D = 40
ship_layout = generate_ship(D)

plt.figure(figsize=(8, 8))
plt.imshow(ship_layout, cmap="gray_r", origin="upper")
plt.title("Generated Ship Layout")
plt.axis("off")
plt.show()

open_cells = np.sum(ship_layout) / (D * D) * 100
print(f"Open Cell Percentage: {open_cells:.2f}%")


# In[ ]:




