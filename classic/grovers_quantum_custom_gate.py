import cirq
import numpy as np

class GroverDiffusionOperator(cirq.Gate):
    def __init__(self, num_qubits):
        super(GroverDiffusionOperator, self)
        self.num_q = num_qubits
        state = [[0.0] for _ in range(2**self.num_q)]
        state[0] = [1.0]
        state = np.array(state)
        self.unitary = 2 * state @ state.T - np.identity(2**self.num_q)
        print(self.unitary)

    def _unitary_(self):
        return self.unitary

    def _num_qubits_(self):
        return self.num_q

    def _circuit_diagram_info_(self, args):
        return ["GD" for _ in range(self.num_q)]

def get_n_qbits(n):
    return cirq.LineQubit.range(n)

# All 1 Oracle
def oracle(bits):
    ret = []
    ret.append(cirq.Z(bits[-1]).controlled_by(*bits[:-1]))
    return ret 

def grover_operation(n):
    return GroverDiffusionOperator(n)

def make_circuit(n, oracle, depth):
    qbits = get_n_qbits(n)
    c = cirq.Circuit()
    c.append([cirq.H(i) for i in qbits])
    for _ in range(depth):
        c.append(oracle(qbits))
        c.append([cirq.H(i) for i in qbits])
        c.append(grover_operation(n).on(*qbits))
        c.append([cirq.H(i) for i in qbits])
    c.append(cirq.measure(*qbits, key='result'))
    return c

def bitstring(bits):
    return ''.join(str(int(b)) for b in bits)

n = 5
rep = 3
circuit = make_circuit(n, oracle, rep)
print(circuit)
simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=1000)
frequencies = result.histogram(key='result', fold_func=bitstring)
print('Sampled results:\n{}'.format(frequencies))
print("Actual result:", n*"1")
