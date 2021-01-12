import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def anzatz(circuit, qubits, parameters):
    for i in range(len(qubits)):
        circuit.append([cirq.ry(parameters[0]).on(qubits[i])])
    circuit.append(cirq.ry(-np.pi/2).on(qubits[0]))
    circuit.append(cirq.rx(np.pi/2).on(qubits[1]))
    return circuit

# H = a * I + b * X + c * Y + d * Z
def hamiltonian(qubits, a, b, c, d):
    h = [a]
    h.append(b * cirq.Z(qubits[0]))
    h.append(c * cirq.Z(qubits[1]))
    h.append(d * cirq.Z(qubits[2]))
    return h

qubits = [cirq.GridQubit(0, i) for i in range(3)]
params = [sympy.symbols('vqe')]
vqe_circuit = anzatz(cirq.Circuit(), qubits, params)
readout_operators = sum(hamiltonian(qubits, 1, 1, 0, 2))
print(readout_operators)
print(vqe_circuit)
inputs = tfq.convert_to_tensor([cirq.Circuit()])

ins = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)
init = tf.keras.initializers.RandomUniform(0, 2 * np.pi)
outs = tfq.layers.PQC(vqe_circuit, readout_operators, repetitions=1000, differentiator=tfq.differentiators.CentralDifference(), initializer=init)(ins)
#outs = tfq.layers.PQC(vqe_circuit, readout_operators, differentiator=tfq.differentiators.CentralDifference(), initializer=init)(ins)
vqe = tf.keras.models.Model(inputs=ins, outputs=outs)
vqe.compile(loss=tf.keras.losses.MAE, optimizer=tf.keras.optimizers.Adam(lr=3e-3))
'''
optimal = np.array([-2])
history = vqe.fit(inputs, optimal, epochs=1000)

plt.plot(history.history['loss'])
plt.title("VQE")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()
print(vqe.trainable_variables)
print(vqe(inputs))
'''
def f(x):
    vqe.set_weights(np.array([x]))
    ret = vqe(inputs)
    return ret.numpy()[0][0]

opt = minimize(f, np.random.uniform(0, 2*np.pi, 1), method='Nelder-Mead')
print(opt)

print(vqe.trainable_variables)
pred = vqe(inputs).numpy()
print(pred)

N = 100
xs = np.linspace(0, 2*np.pi, N)
energies = []
for i in range(N):
    energies.append(f([xs[i]]))

plt.plot(xs, energies)
plt.xlabel("Parameter")
plt.ylabel("Energy/Eigenvalue")
plt.title("VQE")
plt.show()
