p_max = 0
p_tot = 0
for _ in range(10):
    p_o, p_i = map(int, input().split(' '))
    p_tot += p_i - p_o
    if p_tot > p_max:
        p_max = p_tot

print(p_max)