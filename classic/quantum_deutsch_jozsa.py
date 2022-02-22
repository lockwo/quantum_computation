import cirq
import random


def get_n_qbits(n):
    return cirq.LineQubit.range(n)

def balanced_oracle(n, qbits):
    if n == 1:
        return [cirq.CNOT(qbits[0], qbits[1])]
    elif n == 2:
        return [cirq.CNOT(qbits[0], qbits[2]), cirq.CNOT(qbits[1], qbits[2])]

def constant_oralce(n, qbits):
    if n == 1:
        return [cirq.CNOT(qbits[0], qbits[1]), cirq.X(qbits[1]), cirq.CNOT(qbits[0], qbits[1])]
    elif n == 2:
        return [cirq.X(qbits[1])]

def make_circuit(n, oracle):
    qbits = get_n_qbits(max(2, n+1))
    c = cirq.Circuit()
    c.append([cirq.X(qbits[n])])
    c.append([cirq.Moment([cirq.H(i) for i in qbits])])
    c.append(oracle(n, qbits))
    c.append([cirq.Moment([cirq.H(qbits[i]) for i in range(len(qbits) - 1)])])
    c.append(cirq.measure(*qbits[:n], key='result'))
    return c

def bitstring(bits):
    return ''.join(str(int(b)) for b in bits)

N = 2

if random.random() < 0.5:
    oracle = constant_oralce
    correct = "CONSTANT"
else:
    oracle = balanced_oracle
    correct = "BALANCED"

circuit = make_circuit(N, oracle)
print(circuit)
simulator = cirq.Simulator()
if N == 1:
    result = simulator.run(circuit)
    print(result)
    result = int(str(result)[7])
    print(result)
    if result == 1:
        print("BALANCED, actual:", correct)
    else: 
        print("CONSTANT, actual:", correct)
else:
    result = simulator.run(circuit, repetitions=1024)
    frequencies = result.histogram(key='result', fold_func=bitstring)
    print('Sampled results:\n{}'.format(frequencies))
    print("High 0*N bit = constant, actual:", correct)
