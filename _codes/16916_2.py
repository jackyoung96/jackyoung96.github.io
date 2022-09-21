# KMP algorithm 

import sys
input = sys.stdin.readline

S = input().strip()
P = input().strip()

# pi table
def make_table(p):
    table = [0 for _ in range(len(p))]
    j = 0
    for i in range(1,len(p)):
        while j > 0 and p[j] != p[i]:
            j = table[j-1] # recursive (go to the j-1 length pi value)

        if p[i] == p[j]:
            j += 1
            table[i] = j
    
    return table

pi_table = make_table(P)

i = 0
j = 0
for i in range(len(S)):    
    while j > 0 and S[i] != P[j]:
        j = pi_table[j-1] # j-1 !!! 주의 할 것

    if S[i] == P[j]:
        j += 1
        if j == len(P):
            print(1)
            exit()
print(0)