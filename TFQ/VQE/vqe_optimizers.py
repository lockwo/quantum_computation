import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from noisyopt import minimizeSPSA
from rotos import rotosolve

def f(x):
    vqe.set_weights(np.array([x]))
    ret = vqe(tfq.convert_to_tensor([cirq.Circuit()]))
    return ret.numpy()[0][0]

def anzatz(circuit, qubits, parameters):
    for i in range(5):
        pos_up = int(i*2)
        pos_down = pos_up + 1
        circuit.append([cirq.X(qubits[pos_down])])
        circuit.append([cirq.ry(np.pi/2).on(qubits[pos_up])])
        circuit.append([cirq.rx(-np.pi/2).on(qubits[pos_down])])
        circuit.append([cirq.CNOT(qubits[pos_up], qubits[pos_down])])
        circuit.append([cirq.rz(parameters[0]).on(qubits[pos_down])])
        circuit.append([cirq.CNOT(qubits[pos_up], qubits[pos_down])])
        circuit.append([cirq.ry(-np.pi/2).on(qubits[pos_up])])
        circuit.append([cirq.rx(np.pi/2).on(qubits[pos_down])])
    
    circuit.append([cirq.SWAP(qubits[0], qubits[1])])
    circuit.append([cirq.CNOT(qubits[5], qubits[4])])
    circuit.append([cirq.Z(qubits[6]), cirq.Z(qubits[7])])
    circuit.append([cirq.S(qubits[6]), cirq.S(qubits[7])])
    circuit.append([cirq.H(qubits[6]), cirq.H(qubits[7])])
    circuit.append([cirq.CNOT(qubits[7], qubits[6])])
    circuit.append([cirq.H(qubits[8]), cirq.H(qubits[9])])
    circuit.append([cirq.CNOT(qubits[9], qubits[8])])
    return circuit

def hamiltonian(qubits, a, b, c, d, e, f):
    h = [a]
    h.append(b * cirq.Z(qubits[1]))
    h.append(c * cirq.Z(qubits[2]))
    h.append(d * (cirq.Z(qubits[4]) + cirq.Z(qubits[5])))
    h.append(e * (cirq.Z(qubits[6]) + cirq.Z(qubits[7])))
    h.append(f * (cirq.Z(qubits[8]) + cirq.Z(qubits[9])))
    return h   

all_coeff = [
    [2.8489, 0.5678, -1.4508, 0.6799, 0.0791, 0.0791],
    [2.1868, 0.5449, -1.2870, 0.6719, 0.0798, 0.0798],
    [1.1182, 0.4754, -0.9145, 0.6438, 0.0825, 0.0825],
    [0.7381, 0.4325, -0.7355, 0.6233, 0.0846, 0.0846],
    [0.4808, 0.3937, -0.5950, 0.6025, 0.0870, 0.0870],
    [0.2976, 0.3593, -0.4826, 0.5818, 0.0896, 0.0896],
    [0.2252, 0.3435, -0.4347, 0.5716, 0.0910, 0.0910],
    [0.0609, 0.3018, -0.3168, 0.5421, 0.0954, 0.0954], 
    [-0.1253, 0.2374, -0.1603, 0.4892, 0.1050, 0.1050],
    [-0.1927, 0.2048, -0.0929, 0.4588, 0.1116, 0.1116], 
    [-0.2632, 0.1565, -0.0088, 0.4094, 0.1241, 0.1241],
    [-0.2934, 0.1251, 0.0359, 0.3730, 0.1347, 0.1347],
    [-0.3018, 0.1142, 0.0495, 0.3586, 0.1392, 0.1392],
    [-0.3104, 0.1026, 0.0632, 0.3406, 0.1450, 0.1450],
    [-0.3135, 0.0984, 0.0679, 0.3329, 0.1475, 0.1475]
]

dist = [
    0.2, 
    0.25,
    0.4,
    0.5,
    0.6,
    0.7,
    0.75,
    0.9,
    1.2,
    1.4, 
    1.8,
    2.2,
    2.4,
    2.7,
    2.85
]

qubits = [cirq.GridQubit(0, i) for i in range(10)]
params = [sympy.symbols('vqe')]
vqe_circuit = anzatz(cirq.Circuit(), qubits, params)

nms = []
spsas = []
rotos = []
for j in range(20):
    nm = []
    spsa = []
    roto = []
    for i in range(len(all_coeff)):
        print(i, len(all_coeff))
        coeff = all_coeff[i]
        readout_operators = sum(hamiltonian(qubits, coeff[0], coeff[1], coeff[2], coeff[3], coeff[4], coeff[5]))
        ins = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)
        outs = tfq.layers.PQC(vqe_circuit, readout_operators, repetitions=1000)(ins)
        vqe = tf.keras.models.Model(inputs=ins, outputs=outs)
        opt = minimize(f, np.random.uniform(0, 2*np.pi, 1), method='Nelder-Mead')
        nm.append(opt['fun'])
        opt = minimizeSPSA(f, bounds=[[0, 2*np.pi]], x0=np.random.uniform(0, 2*np.pi, 1), niter=200, paired=False)
        spsa.append(opt['fun'])
        opt = rotosolve(f, 1)
        roto.append(opt['fun'])
    nms.append(nm)
    spsas.append(spsa)
    rotos.append(roto)

fig, ax = plt.subplots()

avg = np.mean(np.array(nms), axis=0)
plt.plot(dist, avg, color='red', label='NM')
plus = avg + np.std(avg, axis=0)
minus = avg - np.std(avg, axis=0)
ax.fill_between(dist, (minus), (plus), alpha=0.2, color='red')
ax.fill_between(dist, (minus), (plus), alpha=0.2, color='red')
plt.xlabel("Bond Length")
plt.ylabel("Energy")
plt.legend()
plt.show()
avg = np.mean(np.array(spsas), axis=0)
plt.plot(dist, avg, label='SPSA', color='blue')
plus = avg + np.std(avg, axis=0)
minus = avg - np.std(avg, axis=0)
ax.fill_between(dist, (minus), (plus), alpha=0.2, color='blue')
ax.fill_between(dist, (minus), (plus), alpha=0.2, color='blue')
plt.xlabel("Bond Length")
plt.ylabel("Energy")
plt.legend()
plt.show()
avg = np.mean(np.array(rotos), axis=0)
plt.plot(dist, avg, label='Roto', color='orange')
plus = avg + np.std(avg, axis=0)
minus = avg - np.std(avg, axis=0)
ax.fill_between(dist, (minus), (plus), alpha=0.2, color='orange')
ax.fill_between(dist, (minus), (plus), alpha=0.2, color='orange')
plt.xlabel("Bond Length")
plt.ylabel("Energy")
plt.legend()
plt.show()

avg = np.mean(np.array(nms), axis=0)
plt.plot(dist, avg, color='red', label='NM')
avg = np.mean(np.array(spsas), axis=0)
plt.plot(dist, avg, label='SPSA', color='blue')
avg = np.mean(np.array(rotos), axis=0)
plt.plot(dist, avg, label='Roto', color='orange')
plt.xlabel("Bond Length")
plt.ylabel("Energy")
plt.legend()
plt.show()
