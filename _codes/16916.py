# Boyer Moore algorithm (시간 초과 발생)

import sys
input = sys.stdin.readline

S = input().strip()
P = input().strip()

# automata
automata = {}
for i,c in enumerate(P):
    automata[c] = len(P)-1-i

i = len(P)-1
result = 0
while i < len(S):
    count = 0
    for j in range(0,-len(P),-1):
        if P[j+(len(P)-1)] == S[i+j]:
            count += 1
        else:
            i += automata.get(S[i+j], len(P)-j)+j
            break
    if count == len(P):
        result = 1
        break

print(result)