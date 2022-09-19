T = int(input())
As = []
for _ in range(T):
    As.append(sorted(list(map(int, input().split(' ')))))

for A in As:
    print(A[7])
