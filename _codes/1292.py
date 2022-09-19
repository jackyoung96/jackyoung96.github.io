A = []
i=1
while len(A) <= 1000:
    for _ in range(i):
        A.append(i)
    i += 1
start,end = map(int, input().split(' '))
print(sum(A[start-1:end]))