import sys
from math import factorial as fac
input = sys.stdin.readline

S = input().strip()
count = [0] * 3 # A,B,C count
for s in S:
    for i,c in [(0,'A'), (1,'B'), (2,'C')]:
        if s == c:
            count[i] += 1

# 5중 리스트... [a][b][c][pre1][pre2]
dp = [[[[[-1] * 3 for _ in range(3)] for _ in range(51)] for _ in range(51)] for _ in range(51)]

answer = ""
def solve(a,b,c,pre1,pre2):
    global answer

    if a+b+c == len(S):
        return True

    result = dp[a][b][c][pre1][pre2]
    if result != -1:
        return result
    
    if a+1 <= count[0]:
        if solve(a+1,b,c,0,pre1) == 1:
            answer = "A" + answer
            dp[a][b][c][pre1][pre2] = 1
            return 1
    
    if b+1 <= count[1] and pre1!= 1:
        if solve(a,b+1,c,1,pre1) == 1:
            answer = "B" + answer
            dp[a][b][c][pre1][pre2] = 1
            return 1
    
    if c+1 <= count[2] and pre1!=2 and pre2!=2:
        if solve(a,b,c+1,2,pre1) == 1:
            answer = "C" + answer
            dp[a][b][c][pre1][pre2] = 1
            return 1

    dp[a][b][c][pre1][pre2] = 0
    return 0

solve(0,0,0,0,0)
print(-1) if answer=="" else print(answer)
