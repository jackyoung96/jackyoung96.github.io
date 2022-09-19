n = int(input())
n_1, n_2 = 0,1
result = 1
if n == 0:
    print(0)
else:
    for _ in range(n-1):
        result = n_1 + n_2
        n_1 = n_2
        n_2 = result
    print(result)