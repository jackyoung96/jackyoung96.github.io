import sys
import math
input = sys.stdin.readline
sys.setrecursionlimit(10000000)

N = int(input())
graph = [[] for _ in range(N+1)]

for _ in range(N-1):
    u,v = map(int, input().split())
    graph[u].append(v)
    graph[v].append(u)

dp = [[0,1] for _ in range(N+1)] # initialize by leaf
visited = [0 for _ in range(N+1)]
# dp[i][1]: i가 얼리어답터일 때 최적해
# dp[i][0]: i가 얼리어답터가 아닐 때 최적해
def dfs(u):
    visited[u] = 1
    for v in graph[u]:
        if visited[v] == 0:
            dfs(v)
            dp[u][0] += dp[v][1] # u가 얼리어답터가 아닐 때 -> v는 전부 얼리어답터
            dp[u][1] += min(dp[v]) # u가 얼리어답터일 때 -> v는 상관 없음
dfs(1)
print(min(dp[1]))