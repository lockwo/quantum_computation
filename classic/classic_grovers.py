
def get_binary(n, l):
    if n == 0:
        return l
    else:
        if len(l) == 0:
            return get_binary(n-1, ["0", "1"])
        else:
            return get_binary(n-1, [i + "0" for i in l] + [i + "1" for i in l])

def f(binary):
    if ''.join(binary) == "11111":
        return True
    return False

def classical_grover(l):
    for index, element in enumerate(l):
        if f(element):
            return index

l = get_binary(5, [])

print(classical_grover(l))
