import tensorflow as tf
import tensorflow_quantum as tfq
import cirq 
import sympy
import numpy as np
import matplotlib.pyplot as plt

def create_circuit(qubits, noise, gamma, params):  
    cir = cirq.Circuit()
    counter = 0
    for i in qubits:
        cir += cirq.ry(params[counter]).on(i)
        counter += 1
    cir += cirq.CNOT(qubits[0], qubits[1])
    for i in qubits:
        cir += noise(gamma).on(i)
    cir += cirq.CNOT(qubits[1], qubits[2])
    for i in qubits:
        cir += noise(gamma).on(i)
    cir += cirq.CNOT(qubits[2], qubits[3])
    for i in qubits:
        cir += noise(gamma).on(i)
    for i in qubits:
        cir += cirq.ry(params[counter]).on(i)
        counter += 1
    return cir

def create_model(circuit, ham):
    ins = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)
    outs = tfq.layers.NoisyPQC(circuit, ham, repetitions=500, sample_based=True, differentiator=tfq.differentiators.ParameterShift())(ins)
    vqe = tf.keras.models.Model(inputs=ins, outputs=outs)
    return vqe

def optimize_model(model, optimizer, iter=80, tol=1e-4):
    inputs = tfq.convert_to_tensor([cirq.Circuit()])
    old = 10
    for _ in range(iter):
        with tf.GradientTape() as tape:
            energy = model(inputs)
        grads = tape.gradient(energy, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if abs(old - energy) < tol:
            break
        old = energy
    return energy.numpy()[0][0]


qubit = [cirq.GridQubit(0, i) for i in range(4)]
noises = [cirq.PhaseDampingChannel, cirq.AmplitudeDampingChannel, cirq.DepolarizingChannel]
gammas = np.linspace(0, 1, 40)
ps = sympy.symbols("q0:8")
Z = cirq.Z
X = cirq.X
Y = cirq.Y
hamiltonian = [Z(qubit[0]) * Z(qubit[1]) + 1/2 * (X(qubit[0]) * X(qubit[1]) + Y(qubit[0]) * Y(qubit[1]) + X(qubit[2]) * X(qubit[3]) + Y(qubit[2]) * Y(qubit[3]))]
opt = tf.keras.optimizers.Adam(lr=0.05)


for j, n in enumerate(noises):
    rs = []
    for g in gammas:
        print(j, n, g)
        r = []
        for _ in range(2):
            ci = create_circuit(qubit, n, g, ps)
            m = create_model(ci, hamiltonian)
            r.append(optimize_model(m, opt))
        rs.append(np.mean(r))
    plt.scatter(gammas, rs, s=10, label=str(n))


plt.legend()
plt.show()
