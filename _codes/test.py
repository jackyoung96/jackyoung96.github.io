from collections import deque
import sys
input = sys.stdin.readline

N,M,K = map(int,input().split())
A = [list(map(int,input().split())) for _ in range(N)]
trees = [list(map(int,input().split())) for _ in range(M)]

tree_dict = [[deque() for _ in range(N)] for _ in range(N)]
for x,y,age in trees:
    tree_dict[x-1][y-1].append(age)
trees = tree_dict
feed = [[5]*N for _ in range(N)]

for _ in range(K):
    for x in range(N):
        for y in range(N):
            tree_list = trees[x][y]
            new_list = deque()
            while len(tree_list):
                age = tree_list.popleft()
                if feed[x][y] >= age:
                    feed[x][y] -= age
                    new_list.append(age+1)
                else:
                    tree_list.append(age)
                    break
            feed[x][y] += sum([a//2 for a in tree_list])
            trees[x][y] = new_list

    for x in range(N):
        for y in range(N):
            for age in trees[x][y]:
                if age%5 == 0:
                    for nx,ny in [[x-1,y-1],[x-1,y],[x-1,y+1],[x,y-1],[x,y+1],[x+1,y-1],[x+1,y],[x+1,y+1]]:
                        if 0<=nx<N and 0<=ny<N:
                            trees[nx][ny].appendleft(1)
    
    for i in range(N):
        for j in range(N):
            feed[i][j] += A[i][j]

result = 0
for x in range(N):
    for y in range(N):
        result += len(trees[x][y])
print(result)