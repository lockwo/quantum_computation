import tensorflow as tf
import tensorflow_quantum as tfq
import cirq 
import sympy
import numpy as np
import matplotlib.pyplot as plt

def create_circuit(qubits, version, noise, gamma, params):  
    cir = cirq.Circuit()
    cir += cirq.ry(params[0]).on(qubits[0])
    cir += cirq.ry(params[1]).on(qubits[1])
    cir += cirq.CNOT(qubits[0], qubits[1])
    cir += noise(gamma).on(qubits[0])
    cir += noise(gamma).on(qubits[1])
    if version == "a":
        cir += cirq.ry(params[2]).on(qubits[0])
    elif version == "b":
        cir += cirq.ry(params[2]).on(qubits[1])
    elif version == "c":
        cir += cirq.ry(params[2]).on(qubits[0])
        cir += cirq.ry(params[3]).on(qubits[1])
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

qubit = [cirq.GridQubit(0, i) for i in range(2)]
noises = [cirq.PhaseDampingChannel, cirq.AmplitudeDampingChannel, cirq.DepolarizingChannel]
gammas = np.linspace(0, 1, 40)
circuits = ["a", "b", "c"]
ps = [sympy.symbols("q0:3"), sympy.symbols("q0:3"), sympy.symbols("q0:4")]
hamiltonian = [cirq.Z(qubit[0]) * cirq.Z(qubit[1]) + cirq.X(qubit[0]) + cirq.X(qubit[1])]
opt = tf.keras.optimizers.Adam(lr=0.05)

fig, axs = plt.subplots(1, 3)

for j, n in enumerate(noises):
    for i, c in enumerate(circuits):
        rs = []
        for g in gammas:
            print(j, n, i, c, g)
            r = []
            for _ in range(2):
                ci = create_circuit(qubit, c, n, g, ps[i])
                m = create_model(ci, hamiltonian)
                r.append(optimize_model(m, opt))
            rs.append(np.mean(r))
        axs[j].plot(gammas, rs, label=c)
        axs[j].set_xlabel("Gamma")

axs[0].set_title('Phase damping')
axs[1].set_title('Amplitude damping')
axs[2].set_title('Depolarizing')

axs[0].legend()
axs[0].set_ylabel("Energy")

plt.show()
