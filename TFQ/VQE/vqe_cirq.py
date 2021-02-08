import cirq
import numpy as np
from scipy.optimize import minimize
from noisyopt import minimizeSPSA
from rotos import rotosolve
import random

def layer(circuit, qubits, parameters):
    for i in range(len(qubits)):
        circuit.append([cirq.rx(parameters[0]).on(qubits[i])])
        circuit.append([cirq.rz(parameters[1]).on(qubits[i])])
    for i in range(len(qubits)-1):
        circuit.append([cirq.CNOT(qubits[i], qubits[i+1])])
    return circuit

def ansatz(circuit, qubits, layers, parameters):
    for i in range(layers):
        circuit = layer(circuit, qubits, parameters[2*i:2*(i+1)])
    return circuit

def hamiltonian(circuit, qubits, ham):
    for i in range(len(qubits)):
        if ham[i] == "x":
            circuit.append(cirq.ry(-np.pi/2).on(qubits[i]))
        elif ham[i] == "y":
            circuit.append(cirq.rx(np.pi/2).on(qubits[i]))
    return circuit

def bitstring(bits):
    return ''.join(str(int(b)) for b in bits)

def exp_val(circuit, qubits, num):
    if len(qubits) == 0:
        return 1
    circuit.append(cirq.measure(*qubits, key='result'))
    simulator = cirq.Simulator()
    results = simulator.run(circuit, repetitions=num)
    measure = results.measurements['result']
    expect = 0
    for i in range(num):
        if sum(measure[i]) % 2 == 0:
            expect += 1
        else:
            expect -= 1
    expect /= num
    return expect

def create_vqe(qubits, layers, parameters, ham):
    circuit = ansatz(cirq.Circuit(), qubits, layers, parameters)
    circuit = hamiltonian(circuit, qubits, ham)
    return circuit
    
possibilities = ["i", "x", "y", "z"]
l = 3
q = 2
hamilton = [[random.choice(possibilities) for _ in range(q)] for _ in range(l)]
h_weights = [random.uniform(0, 1) for _ in range(l)]
lay = 2

def vqe(parameters):
    qubits = [cirq.GridQubit(0, i) for i in range(q)]
    expectation = 0
    for i in range(len(h_weights)):
        circuit = create_vqe(qubits, lay, parameters, hamilton[i])
        expectation += (h_weights[i] * exp_val(circuit, [qubits[j] for j in range(len(hamilton[i])) if hamilton[i][j] != "i"], 1000))
    return expectation

def real_min(h, w):
    for i in range(len(h)):
        for j in range(len(h[i])):
            if h[i][j] == "x":
                h[i][j] = np.array([[0, 1], [1, 0]])
            elif h[i][j] == "y":
                h[i][j] = np.array([[0, -1j], [1j, 0]])
            elif h[i][j] == "z":
                h[i][j] = np.array([[1, 0], [0, -1]])
            elif h[i][j] == "i":
                h[i][j] = np.array([[1, 0], [0, 1]])
    full = np.zeros(shape=(2**len(h[0]), 2**len(h[0]))).astype('complex128')
    for i in range(len(h)):
        op = np.kron(h[i][0], h[i][1])
        for j in range(2, len(h[i])):   
            op = np.kron(op, h[i][j])
        full += (w[i] * op)
    eig = np.real(np.linalg.eigvals(full))
    print(full.shape)
    return min(eig)
    

opt = minimize(vqe, np.random.uniform(0, 2*np.pi, q * lay), method='Nelder-Mead')
print("NM", opt['fun'])
opt = minimize(vqe, np.random.uniform(0, 2*np.pi, q * lay), method='COBYLA')
print("COBYLA", opt['fun'])
opt = opt = minimizeSPSA(vqe, bounds=[[0, 2*np.pi] for _ in range(q * lay)], x0=np.random.uniform(0, 2*np.pi, q * lay), niter=200, paired=False)
print("SPSA", opt['fun'])
opt = rotosolve(vqe, q * lay)
print("Rotosolve", opt['fun'])

print("Real min:", real_min(hamilton, h_weights))

