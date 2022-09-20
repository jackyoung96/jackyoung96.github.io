stack = []
input_string = input()

def get_value(input_string):
    for c in input_string:
        if c == '(' or c == '[':
            stack.append(c)
        elif c == ')':
            if len(stack) > 0:
                if stack[-1] == '(':
                    stack.pop(-1)
                    stack.append(2)
                elif len(stack) > 1 and isinstance(stack[-1], int) and stack[-2] == '(':
                    num = stack.pop(-1)
                    stack.pop(-1)
                    stack.append(2*num)
                else:
                    return 0
            else:
                return 0
        if c == ']':
            if len(stack) > 0:
                if stack[-1] == '[':
                    stack.pop(-1)
                    stack.append(3)
                elif len(stack) > 1 and isinstance(stack[-1], int) and stack[-2] == '[':
                    num = stack.pop(-1)
                    stack.pop(-1)
                    stack.append(3*num)
                else:
                    return 0
            else:
                return 0
        
        if len(stack) > 1 and isinstance(stack[-1], int) and isinstance(stack[-2], int):
            num = stack.pop(-1)
            stack[-1] += num
    
    if len(stack) == 1 and isinstance(stack[0], int):
        return stack[0]
    else:
        return 0

print(get_value(input_string))