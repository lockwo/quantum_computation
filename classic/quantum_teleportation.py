import cirq
import numpy as np

def mom(x):
    return [cirq.Moment(x)]

def bell_state(qubit1, qubit2, circuit):
    circuit.append([cirq.H(qubit1)])
    circuit.append([cirq.CNOT(qubit1, qubit2)])

def teleport(qubits, circuit, x, y):
    circuit.append([cirq.X(qubits[0])**x, cirq.Y(qubits[0])**y])
    circuit.append([cirq.CNOT(qubits[0], qubits[1])])
    circuit.append([cirq.H(qubits[0])])
    #circuit.append([cirq.measure(*[qubits[0], qubits[1]], key='Alice')])
    circuit.append([cirq.CZ(qubits[0], qubits[2])])
    circuit.append([cirq.CNOT(qubits[1], qubits[2])])
    #circuit.append([cirq.Moment([cirq.measure(*[qubits[2]], key='Bob')])])
    print(circuit)
    #input()
    simulator = cirq.Simulator(seed=3)
    final = simulator.simulate(circuit)
    print("Bob message:", final.bloch_vector_of(qubits[2]))
    #run(circuit, 'Bob', 1000)

def bitstring(bits):
    return ''.join(str(int(b)) for b in bits)

def run(circuit, k, rep):
    simulator = cirq.Simulator(seed=3)
    result = simulator.run(circuit, repetitions=rep)
    data = np.array(result.data[k])
    frequencies = result.histogram(key=k, fold_func=bitstring)
    print('Sampled results:\n{}'.format(frequencies))
    return data[0]

qubits = [cirq.GridQubit(0, i) for i in range(3)]
alice = qubits[:2]
bob = [qubits[-1]]
x = 0.2
y = 0.4
q0 = cirq.LineQubit(0)
sim = cirq.Simulator(seed=3)
c = cirq.Circuit([cirq.X(q0)**x, cirq.Y(q0)**y])
message = sim.simulate(c)
c.append(cirq.measure(q0, key='test'))
#print(c)
run(c, 'test', 1000)
print("Input message:", message.bloch_vector_of(q0))
circuit = cirq.Circuit()
bell_state(alice[1], bob[0], circuit)
teleport(qubits, circuit, x, y)
