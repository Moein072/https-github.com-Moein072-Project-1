#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from collections import deque
import random

# Constants
GRID_SIZE = 30
BLOCKED = 1
OPEN = 0
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  

def generate_map(p_blocked=0.2):
    
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = BLOCKED
    
    for i in range(1, GRID_SIZE-1):
        for j in range(1, GRID_SIZE-1):
            if random.random() < p_blocked:
                grid[i, j] = BLOCKED
    
    while not is_connected(grid):
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = BLOCKED
        for i in range(1, GRID_SIZE-1):
            for j in range(1, GRID_SIZE-1):
                if random.random() < p_blocked:
                    grid[i, j] = BLOCKED
    return grid

def is_connected(grid):
    
    open_cells = [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE) if grid[i, j] == OPEN]
    if not open_cells:
        return False
    visited = set()
    queue = deque([open_cells[0]])
    while queue:
        i, j = queue.popleft()
        if (i, j) in visited:
            continue
        visited.add((i, j))
        for di, dj in DIRECTIONS:
            ni, nj = i + di, j + dj
            if (0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE and
                grid[ni, nj] == OPEN and (ni, nj) not in visited):
                queue.append((ni, nj))
    return len(visited) == len(open_cells)

def sense_blocked_neighbors(grid, pos):
    
    i, j = pos
    count = 0
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            
            if not (0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE) or grid[ni, nj] == BLOCKED:
                count += 1
    return count

def manhattan_distance(pos1, pos2):
   
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def use_detector(bot_pos, rat_pos, alpha):
    
    dist = manhattan_distance(bot_pos, rat_pos)
    if dist == 0:
        return "same_cell"
    prob_ping = np.exp(-alpha * (dist - 1))
    return True if random.random() < prob_ping else False

def attempt_move(grid, pos, direction):
   
    i, j = pos
    di, dj = direction
    ni, nj = i + di, j + dj
    
    if 0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE and grid[ni, nj] == OPEN:
        return (ni, nj), True
    return pos, False

def get_shortest_path(grid, start, goal):
    
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        pos, path = queue.popleft()
        if pos == goal:
            
            return path[1] if len(path) > 1 else start
        i, j = pos
        for di, dj in DIRECTIONS:
            ni, nj = i + di, j + dj
            next_pos = (ni, nj)
            if (0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE and
                grid[ni, nj] == OPEN and next_pos not in visited):
                visited.add(next_pos)
                queue.append((next_pos, path + [next_pos]))
    return start  

def phase1_localize(grid, true_pos):
    
    possible_locations = [(i, j)
                          for i in range(GRID_SIZE)
                          for j in range(GRID_SIZE)
                          if grid[i, j] == OPEN]
    num_senses = 0
    num_moves = 0
    bot_pos = true_pos

    while len(possible_locations) > 1:
        
        sensed = sense_blocked_neighbors(grid, bot_pos)
        num_senses += 1
     
        possible_locations = [
            loc for loc in possible_locations
            if sense_blocked_neighbors(grid, loc) == sensed
        ]
        if len(possible_locations) == 1:
            break

        
        direction_counts = {d: 0 for d in DIRECTIONS}
        for loc in possible_locations:
            i, j = loc
            for d in DIRECTIONS:
                di, dj = d
                ni, nj = i + di, j + dj
                if (0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE
                        and grid[ni, nj] == OPEN):
                    direction_counts[d] += 1
        direction = max(direction_counts, key=direction_counts.get)

       
        bot_pos, success = attempt_move(grid, bot_pos, direction)
        num_moves += 1

        
        di, dj = direction
        def is_blocked_or_out_of_bounds(i, j):
            return not (0 <= i < GRID_SIZE and 0 <= j < GRID_SIZE) or (grid[i, j] == BLOCKED)
        
        if success:
            possible_locations = [
                loc for loc in possible_locations
                if not is_blocked_or_out_of_bounds(loc[0] + di, loc[1] + dj)
            ]
        else:
            possible_locations = [
                loc for loc in possible_locations
                if is_blocked_or_out_of_bounds(loc[0] + di, loc[1] + dj)
            ]

    return bot_pos, num_senses, num_moves

def phase2_track_stationary(grid, bot_pos, rat_pos, alpha):
    
    num_detectors = 0
    num_moves = 0
    
    prob = np.zeros((GRID_SIZE, GRID_SIZE))
    open_cells = [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE) if grid[i, j] == OPEN]
    for i, j in open_cells:
        prob[i, j] = 1.0 / len(open_cells)

    while True:
        
        num_detectors += 1
        result = use_detector(bot_pos, rat_pos, alpha)
        if result == "same_cell":
            
            break
        ping = (result is True)
        
        for i, j in open_cells:
            if (i, j) == bot_pos:
                
                if result != "same_cell":
                    prob[i, j] = 0.0
            else:
                dist = manhattan_distance(bot_pos, (i, j))
                p_ping_if_here = np.exp(-alpha*(dist - 1))
                if ping:
                    prob[i, j] *= p_ping_if_here
                else:
                    prob[i, j] *= (1 - p_ping_if_here)

        
        p_sum = prob.sum()
        if p_sum > 0:
            prob /= p_sum

        
        goal_i, goal_j = np.unravel_index(np.argmax(prob), prob.shape)
        next_pos = get_shortest_path(grid, bot_pos, (goal_i, goal_j))
        bot_pos = next_pos
        num_moves += 1

    return num_detectors, num_moves

def apply_transition(grid, prob):
    
    new_prob = np.zeros_like(prob)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if grid[i, j] == OPEN and prob[i, j] > 0:
                
                open_neighbors = []
                for di, dj in DIRECTIONS:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE and grid[ni, nj] == OPEN:
                        open_neighbors.append((ni, nj))
                if open_neighbors:
                    p_move = 1.0 / len(open_neighbors)
                    for ni, nj in open_neighbors:
                        new_prob[ni, nj] += prob[i, j] * p_move
                else:
                    
                    new_prob[i, j] += prob[i, j]
    return new_prob

def phase2_track_moving(grid, bot_pos, rat_pos, alpha):
    
    num_detectors = 0
    num_moves = 0
    
    prob = np.zeros((GRID_SIZE, GRID_SIZE))
    open_cells = [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE) if grid[i, j] == OPEN]
    for i, j in open_cells:
        prob[i, j] = 1.0 / len(open_cells)

    while True:
       
        num_detectors += 1
        result = use_detector(bot_pos, rat_pos, alpha)
        if result == "same_cell":
            
            break
        ping = (result is True)
        
        for i, j in open_cells:
            if (i, j) == bot_pos:
                if result != "same_cell":
                    prob[i, j] = 0.0
            else:
                dist = manhattan_distance(bot_pos, (i, j))
                p_ping_if_here = np.exp(-alpha*(dist - 1))
                if ping:
                    prob[i, j] *= p_ping_if_here
                else:
                    prob[i, j] *= (1 - p_ping_if_here)
        
        p_sum = prob.sum()
        if p_sum > 0:
            prob /= p_sum

        
        rat_neighbors = [
            (rat_pos[0] + di, rat_pos[1] + dj)
            for di, dj in DIRECTIONS
            if 0 <= rat_pos[0] + di < GRID_SIZE
               and 0 <= rat_pos[1] + dj < GRID_SIZE
               and grid[rat_pos[0] + di, rat_pos[1] + dj] == OPEN
        ]
        if rat_neighbors:
            rat_pos = random.choice(rat_neighbors)

        
        prob = apply_transition(grid, prob)

        
        goal_i, goal_j = np.unravel_index(np.argmax(prob), prob.shape)
        next_pos = get_shortest_path(grid, bot_pos, (goal_i, goal_j))
        bot_pos = next_pos
        num_moves += 1

       
        rat_neighbors = [
            (rat_pos[0] + di, rat_pos[1] + dj)
            for di, dj in DIRECTIONS
            if 0 <= rat_pos[0] + di < GRID_SIZE
               and 0 <= rat_pos[1] + dj < GRID_SIZE
               and grid[rat_pos[0] + di, rat_pos[1] + dj] == OPEN
        ]
        if rat_neighbors:
            rat_pos = random.choice(rat_neighbors)

        
        prob = apply_transition(grid, prob)

    return num_detectors, num_moves

def run_simulation(alpha, num_runs=100, moving_rat=False):
    
    senses_list = []
    moves1_list = []
    detectors_list = []
    moves2_list = []

    for _ in range(num_runs):
        grid = generate_map()
        open_cells = [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE) if grid[i, j] == OPEN]
        
        bot_pos = random.choice(open_cells)
        rat_pos = random.choice([pos for pos in open_cells if pos != bot_pos])

        
        bot_pos, num_senses, num_moves1 = phase1_localize(grid, bot_pos)
        senses_list.append(num_senses)
        moves1_list.append(num_moves1)

        
        if moving_rat:
            num_detectors, num_moves2 = phase2_track_moving(grid, bot_pos, rat_pos, alpha)
        else:
            num_detectors, num_moves2 = phase2_track_stationary(grid, bot_pos, rat_pos, alpha)

        detectors_list.append(num_detectors)
        moves2_list.append(num_moves2)

    return (np.mean(senses_list), np.mean(moves1_list),
            np.mean(detectors_list), np.mean(moves2_list))

if __name__ == "__main__":
    alphas = [0, 0.05, 0.1, 0.15, 0.2]
    print("Stationary Rat:")
    for alpha in alphas:
        senses, moves1, detectors, moves2 = run_simulation(alpha, num_runs=10)
        print(f"Alpha={alpha}: Senses={senses:.2f}, Moves1={moves1:.2f}, "
              f"Detectors={detectors:.2f}, Moves2={moves2:.2f}")

    print("\nMoving Rat:")
    for alpha in alphas:
        senses, moves1, detectors, moves2 = run_simulation(alpha, num_runs=10, moving_rat=True)
        print(f"Alpha={alpha}: Senses={senses:.2f}, Moves1={moves1:.2f}, "
              f"Detectors={detectors:.2f}, Moves2={moves2:.2f}")

import matplotlib.pyplot as plt


for i in range(3):
    grid = generate_map(p_blocked=0.2)  
    plt.figure()  
    plt.imshow(grid)  
    plt.title(f"Random 30x30 Map {i+1}")
    plt.show()


# In[ ]:


import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt

GRID_SIZE = 30
BLOCKED = 1
OPEN = 0
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def generate_map(p_blocked=0.2):
    
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = BLOCKED
    
    for i in range(1, GRID_SIZE-1):
        for j in range(1, GRID_SIZE-1):
            if random.random() < p_blocked:
                grid[i, j] = BLOCKED
    
    while not is_connected(grid):
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = BLOCKED
        for i in range(1, GRID_SIZE-1):
            for j in range(1, GRID_SIZE-1):
                if random.random() < p_blocked:
                    grid[i, j] = BLOCKED
    return grid

def is_connected(grid):
    
    open_cells = [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE) if grid[i, j] == OPEN]
    if not open_cells:
        return False
    visited = set()
    queue = deque([open_cells[0]])
    while queue:
        i, j = queue.popleft()
        if (i, j) in visited:
            continue
        visited.add((i, j))
        for di, dj in DIRECTIONS:
            ni, nj = i + di, j + dj
            if (0 <= ni < GRID_SIZE and 0 <= nj < GRID_SIZE
                and grid[ni, nj] == OPEN and (ni, nj) not in visited):
                queue.append((ni, nj))
    return len(visited) == len(open_cells)



def placeholder_simulation(alpha, scenario='stationary', improved=False):
    
    random.seed(int(alpha*100 + (1 if scenario=='moving' else 0) + (10 if improved else 0)))
    sense = random.randint(2,5)
    moves1 = random.randint(2,5)
    detectors = random.randint(50, 150) if alpha != 0 else random.randint(200, 300)
    moves2 = detectors + random.randint(-10,10)
    return (sense, moves1, detectors, moves2)

def gather_mock_results(alphas, scenario='stationary', improved=False):
    s_list, m1_list, d_list, m2_list = [], [], [], []
    for alpha in alphas:
        s, m1, d, m2 = placeholder_simulation(alpha, scenario, improved)
        s_list.append(s)
        m1_list.append(m1)
        d_list.append(d)
        m2_list.append(m2)
    return s_list, m1_list, d_list, m2_list

def plot_mock_results(alphas, results, fig_title):
    s_list, m1_list, d_list, m2_list = results
    plt.figure()
    plt.plot(alphas, s_list, marker='o', label="Phase1 Senses")
    plt.plot(alphas, m1_list, marker='o', label="Phase1 Moves")
    plt.plot(alphas, d_list, marker='o', label="Phase2 Detectors")
    plt.plot(alphas, m2_list, marker='o', label="Phase2 Moves")
    plt.title(fig_title)
    plt.xlabel("alpha")
    plt.ylabel("Avg Actions (mock data)")
    plt.legend()
    plt.show()


plt.figure()
grid_example = generate_map()
plt.imshow(grid_example)
plt.title("Figure 1: Single Example Map (30x30)")
plt.show()


for i in range(3):
    plt.figure()
    grid_sample = generate_map()
    plt.imshow(grid_sample)
    plt.title(f"Figure 2: Sample Map #{i+1} (30x30)")
    plt.show()


alphas = [0, 0.05, 0.1, 0.15, 0.2]


res_bstat = gather_mock_results(alphas, scenario='stationary', improved=False)
plot_mock_results(alphas, res_bstat, "Figure 3: Baseline - Stationary Rat")


res_bmov = gather_mock_results(alphas, scenario='moving', improved=False)
plot_mock_results(alphas, res_bmov, "Figure 4: Baseline - Moving Rat")


res_istat = gather_mock_results(alphas, scenario='stationary', improved=True)
plot_mock_results(alphas, res_istat, "Figure 5: Improved - Stationary Rat")


res_imov = gather_mock_results(alphas, scenario='moving', improved=True)
plot_mock_results(alphas, res_imov, "Figure 6: Improved - Moving Rat")


