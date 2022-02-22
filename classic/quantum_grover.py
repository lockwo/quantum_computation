import cirq
import math

def get_n_qbits(n):
    return cirq.LineQubit.range(n)

# THIS ORACLE IS TRUE FOR all 1s
def oracle(n, bits):
    ret = []
    ret.append(cirq.Z(bits[n-1]).controlled_by(*bits[:n-1]))
    return ret 

def grover_operation(bits, n):
    ret = []
    ret.append([cirq.Moment([cirq.H(i) for i in bits])])
    ret.append([cirq.Moment([cirq.X(i) for i in bits])])
    ret.append(cirq.H(bits[n-1]))
    ret.append(cirq.X(bits[n-1]).controlled_by(*bits[:n-1]))
    ret.append(cirq.H(bits[n-1]))
    ret.append([cirq.Moment([cirq.X(i) for i in bits])])
    ret.append([cirq.Moment([cirq.H(i) for i in bits])])
    return ret

def make_circuit(n, oracle):
    qbits = get_n_qbits(n)
    c = cirq.Circuit()
    c.append([cirq.H(i) for i in qbits])
    c.append(oracle(n, qbits))
    c.append(grover_operation(qbits, n))
    c.append(cirq.measure(*qbits[:n], key='result'))
    return c

def bitstring(bits):
    return ''.join(str(int(b)) for b in bits)


n = 6
circuit = make_circuit(n, oracle)
print(circuit)
simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=4096)
frequencies = result.histogram(key='result', fold_func=bitstring)
print('Sampled results:\n{}'.format(frequencies))
print("Actual result:", n*"1")
