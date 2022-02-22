import cirq
import time

def get_n_qbits(n):
    return cirq.LineQubit.range(n)

def oracle(n, bits):
    ret = []
    for i in range(n):
        for j in range(n, 2*n):
            ret.append(cirq.CNOT(bits[i], bits[j]))
    return ret

def make_circuit(n, oracle):
    bits = get_n_qbits(2*n)
    c = cirq.Circuit()
    c.append([cirq.Moment([cirq.H(bits[i]) for i in range(n)])])
    c.append(oracle(n, bits))
    c.append([cirq.Moment([cirq.H(bits[i]) for i in range(n)])])
    c.append(cirq.measure(*bits[:n], key='result'))
    return c

def bitstring(bits):
    return ''.join(str(int(b)) for b in bits)

if __name__ == "__main__":
    start_time = time.time()
    n = 6
    circuit = make_circuit(n, oracle)
    print(circuit)
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=64)
    frequencies = result.histogram(key='result', fold_func=bitstring)
    print('Sampled results:\n{}'.format(frequencies))
    print("Actual result:", "1"*n)
    end_time = time.time()
    print(end_time - start_time)
