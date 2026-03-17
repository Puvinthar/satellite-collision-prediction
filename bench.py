import numpy as np
import time
N = 100
T = 100
trajs = np.random.rand(N, T, 3)
start = time.time()
pairs = []
for i in range(N):
    for j in range(i+1, N):
        dist = np.min(np.linalg.norm(trajs[i] - trajs[j], axis=1))
        if dist < 1000:
            pairs.append((i,j,dist))
print(f"Brute force {N} items: {time.time()-start:.3f}s")
