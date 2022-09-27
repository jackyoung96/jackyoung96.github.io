import sys
input = sys.stdin.readline
from collections import deque

N,M,K = map(int, input().split())

# build map
maps = [[1 for _ in range(M)] for _ in range(N)]
visited = [[-1 for _ in range(M)] for _ in range(N)]
for n in range(N):
    row = input().strip()
    for m in range(M):
        if row[m] == '#':
            maps[n][m] = 0

x1,y1,x2,y2 = map(int, input().split())
x1,y1,x2,y2 = map(lambda x: x-1, [x1,y1,x2,y2])

queue = deque([[x1,y1]])
visited[x1][y1] = 0

while queue:
    x,y = queue.popleft()
    if x == x2 and y == y2:
        print(visited[x][y])
        exit()
    
    for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
        for k in range(1, K+1):
            x3, y3 = x+dx*k, y+dy*k
            if (0 <= x3 < N) and (0 <= y3 < M):
                if maps[x3][y3] == 0:
                    break
                elif visited[x3][y3] == -1:
                    visited[x3][y3] = visited[x][y] + 1
                    queue.append([x3,y3])
                elif visited[x3][y3] <= visited[x][y]: # O(NMK) -> O(NM)
                    break
            else:
                break

print(visited[x2][y2])