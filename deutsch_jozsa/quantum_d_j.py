import cirq
import random


def get_n_qbits(n):
    return cirq.LineQubit.range(n)

def balanced_oracle(q1, q2, n):
    return [cirq.CNOT(q1, q2)]


def constant_oralce(q1, q2, n):
    return [cirq.CNOT(q1, q2), cirq.X(q2), cirq.CNOT(q1, q2)]

def make_circuit(n, oracle):
    qbits = get_n_qbits(n+1)
    c = cirq.Circuit()
    c.append([cirq.X(qbits[n])])
    c.append([cirq.Moment([cirq.H(i) for i in qbits])])
    c.append(oracle(qbits[0], qbits[1], n))
    for i in range(n):
        c.append([cirq.H(qbits[i]), cirq.measure(qbits[i], key='result')])
    return c


N = 1 

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
    print("NOT YET IMPLEMENTED")
