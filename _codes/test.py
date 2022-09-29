import sys
input = sys.stdin.readline

total_n = int(input())
customers = list(map(int, input().split()))
small_n = int(input())

cum = [0]
total = 0
for n in range(total_n):
    total+=customers[n]
    cum.append(total)

dp = [[0] * 3 for _ in range(total_n+1)]
for n in range(1,total_n+1):
    if n+small_n-1 < total_n+1:
        dp[n][0] = max(dp[n-1][0], cum[n+small_n-1]-cum[n-1])

        if n-small_n > 0:
            dp[n][1] = max(dp[n-1][1], cum[n+small_n-1]-cum[n-1] + dp[n-small_n][0])
            dp[n][2] = max(dp[n-1][2], cum[n+small_n-1]-cum[n-1] + dp[n-small_n][1])
        else:
            dp[n][1] = dp[n-1][1]
            dp[n][2] = dp[n-1][2]
    else:
        dp[n][0] = dp[n-1][0]

print(dp[total_n-small_n+1][2])
