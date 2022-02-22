import random

def constant(inputs):
    return 1

def balanced(inputs):
    return sum(inputs) % 2

def get_binary(n, l):
    if n == 0:
        return l
    else:
        if len(l) == 0:
            return get_binary(n-1, ["0", "1"])
        else:
            return get_binary(n-1, [i + "0" for i in l] + [i + "1" for i in l])

def get_bits(n):
    strings = get_binary(n, [])
    ret = [[int(i) for i in j] for j in strings]
    return ret

def deutsch_jozsa_classical(inputs, f):
    init = f(inputs[0])
    for i in range(1, len(inputs)//2+1):
        if f(inputs[i]) != init:
            return 1
    return 0


N = 9 # NUMBER OF BITS
inputs = get_bits(N)


if random.random() < 0.5:
    f = constant
    correct = "CONSTANT"
else:
    f = balanced
    correct = "BALANCED"

result = deutsch_jozsa_classical(inputs, f)
if result == 1:
    print("BALANCED, actual:", correct)
else: 
    print("CONSTANT, actual:", correct)
