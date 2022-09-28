import sys
input = sys.stdin.readline

N = int(input())

dp = [i for i in range(N+1)]

for i in range(6,N+1):
    dp[i] = max(2*dp[i-3], 3*dp[i-4], 4*dp[i-5])

print(dp[N])