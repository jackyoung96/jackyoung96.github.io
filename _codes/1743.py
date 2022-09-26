import sys
sys.setrecursionlimit(10000)
input = sys.stdin.readline

N,M,K = map(int, input().split())
maps = [[0 for _ in range(M)] for _ in range(N)]
for _ in range(K):
    r,c = map(int, input().split())
    maps[r-1][c-1] = 1

visited = [[0 for _ in range(M)] for _ in range(N)]

def dfs(i,j,maps):
    global visited, N, M
    result = 1
    visited[i][j] = 1
    for u,v in [(i-1,j), (i+1,j), (i,j-1), (i,j+1)]:
        if u >= 0 and u < N and v >= 0 and v < M:
            if visited[u][v] == 0 and maps[u][v] == 1:
                result += dfs(u,v,maps)

    return result


max_size = 0
for n in range(N):
    for m in range(M):
        if visited[n][m] == 0 and maps[n][m] == 1:
            max_size = max(max_size, dfs(n,m,maps))

print(max_size)