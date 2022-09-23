import sys
input = sys.stdin.readline

n,k = map(int, input().split())

values = [0] + [int(input()) for _ in range(n)]

dp = [0 for _ in range(k+1)]

for i in range(1,n+1):
    for j in range(1,k+1):
        coin = values[i]
        if i == 1:
            dp[j] = 1 if j%coin==0 else 0
        elif j < coin:
            pass
        elif j == coin:
            dp[j] = dp[j] + 1
        else:
            dp[j] = dp[j] + dp[j-coin]

print(dp[k])