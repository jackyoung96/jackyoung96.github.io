import sys
input = sys.stdin.readline

N,M = map(int, input().split())

graph_tall = [[] for _ in range(N+1)]
graph_small = [[] for _ in range(N+1)]
graph_count = [0 for _ in range(N+1)]

for _ in range(M):
    a,b = map(int, input().split())
    graph_small[a].append(b)
    graph_count[b] += 1 # in degree

queue = []
for i in range(1,N+1):
    if graph_count[i] == 0:
        queue.append(i)

result = []
while queue:
    tallest = queue.pop(0)
    result.append(tallest)
    for node in graph_small[tallest]:
        if graph_count[node] > 1:
            graph_count[node] -= 1
        elif graph_count[node] == 1:
            queue.append(node)
            graph_count[node] = 0

print(*result, sep=' ')