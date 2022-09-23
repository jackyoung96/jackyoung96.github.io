from itertools import combinations
N = int(input())

num = []
for i in range(1,11):
    for comb in combinations(range(10),i):
        comb = list(comb)
        comb = sorted(comb, reverse=True)
        num.append(int("".join(map(str,comb))))

num = sorted(num)
if len(num)-1 < N:
    print(-1)
else:
    print(num[N])