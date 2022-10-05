N = int(input())

op_table = {
    'ADD': '0000',
    'SUB': '0001',
    'MOV': '0010',
    'AND': '0011',
    'OR': '0100',
    'NOT': '0101',
    'MULT': '0110',
    'LSFTL': '0111',
    'LSFTR': '1000',
    'ASFTR': '1001',
    'RL': '1010',
    'RR': '1011',
}

def intToBit(a, bits):
    result = ""
    for i in reversed(range(bits)):
        if a // int(2**i) > 0:
            result += '1'
            a %= int(2**i)
        else:
            result += '0'
    return result

for _ in range(N):
    comm = input().split()
    op = comm[0]
    rd,ra,rb = map(int, comm[1:])

    result = ""

    if op[-1] == 'C':
        result += op_table[op[:-1]] + "10"
    else:
        result += op_table[op] + "00"

    result += format(rd,"03b")

    if op in ['MOV', 'MOVC', 'NOT']:
        result += "000"
    else:
        result += format(ra,"03b")
    
    if op[-1] == 'C':
        result += format(rb,"04b")
    else:
        result += format(rb,"03b") + "0"
    
    print(result)