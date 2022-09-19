hs = []
for _ in range(9):
    hs.append(int(input()))

total_h = sum(hs)
hs = sorted(hs)

for i in range(9):
    for j in range(i+1,9):
        if hs[i] + hs[j] == total_h - 100:
            for k in range(9):
                if k!= i and k!=j:
                    print(hs[k])
            exit()