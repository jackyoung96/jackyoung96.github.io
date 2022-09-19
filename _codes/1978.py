N = int(input())
ns = list(map(int, input().split(' ')))

all_n = [i for i in range(2,1001)]
i = 0
while i < len(all_n)-1:
    num_div = all_n[i]
    j = i+1
    while j < len(all_n):
        if all_n[j] % num_div == 0:
            all_n.remove(all_n[j])
        else:
            j+=1
    i+=1

count = 0
for n in ns:
    if n in all_n:
        count += 1

print(count)