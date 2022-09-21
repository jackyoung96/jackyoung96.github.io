from re import L
import sys
from heapq import heappush, heappop
input = sys.stdin.readline
V,E = map(int, input().split())

# build graph
graph = [[] for k in range(V+1)]
for e in range(E):
    u,v,w = map(int, input().split())
    graph[u].append([v,w])
    graph[v].append([u,w])

visit = [0 for _ in range(V+1)]
heap = []
heappush(heap, [0,1]) # start from node 1
result = 0
while len(heap):
    weight, cur_node = heappop(heap)
    if visit[cur_node] == 1:
        continue

    visit[cur_node] = 1
    result += weight
    for v,w in graph[cur_node]:
        if visit[v] == 0:
            heappush(heap, [w, v])

print(result)