import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def anzatz(circuit, qubits, parameters):
    for i in range(len(qubits)):
        circuit.append([cirq.rx(parameters[0]).on(qubits[i])])
        circuit.append([cirq.ry(parameters[1]).on(qubits[i])])

    circuit.append([cirq.H(qubits[1])])
    circuit.append([cirq.CNOT(qubits[0], qubits[1])])
    circuit.append([cirq.CNOT(qubits[2], qubits[3])])
    return circuit

def hamiltonian(qubits, a, b, c):
    h = [a]
    h.append(b * (cirq.Z(qubits[0]) + cirq.Z(qubits[1])))
    h.append(c * (cirq.Z(qubits[2]) + cirq.Z(qubits[3])))
    return h

qubits = [cirq.GridQubit(0, i) for i in range(4)]
params = sympy.symbols('vqe0:2')
vqe_circuit = anzatz(cirq.Circuit(), qubits, params)
readout_operators = sum(hamiltonian(qubits, 0, 1, 1))
print(readout_operators)
print(vqe_circuit)
inputs = tfq.convert_to_tensor([cirq.Circuit()])

ins = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)
outs = tfq.layers.PQC(vqe_circuit, readout_operators)(ins)
vqe = tf.keras.models.Model(inputs=ins, outputs=outs)

def f(x):
    vqe.set_weights(np.array([x]))
    ret = vqe(inputs)
    return ret.numpy()[0][0]

opt = minimize(f, np.random.uniform(0, 2*np.pi, 2), method='Nelder-Mead')
print(opt)
