N = int(input())
ns = list(map(int, input().split(' ')))
if len(ns) != N:
    raise "please matching number of inputs"

minimum, maximum = ns[0],ns[0]
for n in ns:
    if minimum > n:
        minimum = n
    if maximum < n:
        maximum = n
print(minimum, maximum)