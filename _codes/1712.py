A,B,C = map(int,input().split(' '))
# A + n*B < n*C 를 만족하는 n
if C <= B: print(-1)
else: print(A // (C-B) + 1)