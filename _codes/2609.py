A,B = sorted(list(map(int, input().split(' '))))
a,b = A,B
gcf, lcm = A,B

# gcf
c = b
while c > 1:
    c = b % a
    if c == 0:
        gcf = a
    elif c == 1:
        gcf = 1
    else:
        b = a
        a = c

print(gcf)
print(int(A*B/gcf))