import sys
input = sys.stdin.readline

N,M = map(int, input().split())
maps = [[] for _ in range(N)]
for n in range(N):
    row = list(input().strip())
    maps[n] += list(map(int, row))

# BFS
queue = [(0,0,1)]
visit = [[0 for _ in range(M)] for _ in range(N)]
visit[0][0] = 1
while queue:
    i,j,w = queue.pop(0)
    if i > 0:
        if maps[i-1][j] == 1 and visit[i-1][j] == 0:
            queue.append((i-1,j,w+1))
            visit[i-1][j] = 1
    if i < N-1:
        if maps[i+1][j] == 1 and visit[i+1][j] == 0:
            queue.append((i+1,j,w+1))
            visit[i+1][j] = 1
    if j > 0:
        if maps[i][j-1] == 1 and visit[i][j-1] == 0:
            queue.append((i,j-1,w+1))
            visit[i][j-1] = 1
    if j < M-1:
        if maps[i][j+1] == 1 and visit[i][j+1] == 0:
            queue.append((i,j+1,w+1))
            visit[i][j+1] = 1
    
    if i==N-1 and j==M-1:
        print(w)
        exit()