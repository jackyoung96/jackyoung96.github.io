N,K = map(int, input().split(' '))
k = 0
for n in range(1,N+1):
    if N % n == 0:
        k += 1
    if k == K:
        print(n)
        break
if k < K:
    print(0)