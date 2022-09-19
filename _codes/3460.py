T = int(input())
for _ in range(T):
    n = int(input())
    result = []
    for i in range(20,-1,-1):
        if n >= (2**i):
            result.append(i)
            n -= 2**i
        if n==0:
            break
    print(" ".join(map(str,reversed(result))))
