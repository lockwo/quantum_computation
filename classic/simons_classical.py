import random
import time

# This code randomly generates the function, so queries it 2^n times which is suboptimal, but conveys the idea. 
# It prints out the function table and the solution


def decimalToBinaryFixLength(_length, _decimal):
	binNum = bin(int(_decimal))[2:]
	outputNum = [int(item) for item in binNum]
	if len(outputNum) < _length:
		outputNum = [0]*(_length-len(outputNum)) + outputNum
	return [int(i) for i in outputNum]

def equals(arr1, arr2):
    for i in range(len(arr1)):
        if arr1[i] != arr2[i]:
            return False
    return True

def oracle_function(n):
    table = [[random.randint(0, 1) for i in range(n)]]
    next_ = [i for i in range(2**(n-1))]
    random.shuffle(next_)
    for i in range(1, 2**(n-1)):
        test = [random.randint(0, 1) for i in range(n)]
        while True:
            flag = True
            for i in range(len(table)):
                if equals(test, table[i]):
                    test = [random.randint(0, 1) for i in range(n)]
                    flag = False
                    break
            if flag:
                break
        table.append(test)
    for i in range(2**(n-1)):
        table.append(table[next_[i]])
    return table
            

def solve(n):
    i = 1
    f = oracle_function(n)
    #print("TABLE", f)
    test = [decimalToBinaryFixLength(n, 0), f[0]]
    #print(test)
    while i < 2**n:
        binary = decimalToBinaryFixLength(n, i)
        if equals(f[i], test[1]):
            return [test[0][i] ^ binary[i] for i in range(n)]
        i += 1
         

# This not not the most optimal solution, this is simply an easy one to code that is approximate of the worst case 
if __name__ == "__main__":
    start_time = time.time()
    print(solve(14))
    end_time = time.time()
    print(end_time - start_time)
