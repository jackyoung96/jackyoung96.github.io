M = int(input())
N = int(input())

all_n = [i for i in range(2,N+1)]
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

while len(all_n)!=0 and all_n[0] < M:
    all_n.pop(0)

if len(all_n) == 0:
    print(-1)
else:
    print(sum(all_n))
    print(all_n[0])