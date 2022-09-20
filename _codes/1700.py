N,K = map(int, input().split(' '))
elecs = list(map(int, input().split(' ')))

count = 0
multitap = []
for i, elec in enumerate(elecs):
    if elec in multitap:
        continue
    elif len(multitap) < N:
        multitap.append(elec)
    else:
        multitap_copy = [x for x in multitap]
        for e in elecs[i:]:
            if e in multitap_copy:
                multitap_copy.remove(e)
            if len(multitap_copy)==1:
                break
        multitap.remove(multitap_copy[0])
        multitap.append(elec)
        count += 1
    
print(count)