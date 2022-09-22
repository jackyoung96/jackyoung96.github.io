import math
S = int(input())

# N*(N-1)/2 < S < (N+1)*N/2 -> N-1개
N = int((1+math.sqrt(1+8*S))/2) - 1
print(N)