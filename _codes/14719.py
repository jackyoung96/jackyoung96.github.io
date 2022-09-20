H,W = map(int, input().split(' '))

blocks = list(map(int, input().split(' ')))

max_idx, max_h = 0,0
for idx, h in enumerate(blocks):
    if h > max_h:
        max_h = h
        max_idx = idx

total = 0

# from left
h_t = 0
for w in range(max_idx):
    h = blocks[w]
    total += max(h_t-h,0)
    if h_t < h:
        h_t = h

# from right
h_t = 0
for w in range(W-1,max_idx,-1):
    h = blocks[w]
    total += max(h_t-h,0)
    if h_t < h:
        h_t = h

print(total)