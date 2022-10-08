import math

N = int(input())
studs = list(map(int, input().split()))
B,C = map(int, input().split())

result = 0
for stud in studs:
    if stud <= B:
        result += 1
    else:
        result += 1 + math.ceil((stud-B)/C)

print(result)