import tensorflow as tf
import tensorflow_quantum as tfq
import cirq 
import sympy
import numpy as np
import matplotlib.pyplot as plt

qubit = cirq.GridQubit(0, 0)

params = sympy.symbols("q0:2")
circuit = cirq.Circuit()
circuit += cirq.ry(params[0]).on(qubit)
circuit += cirq.DepolarizingChannel(0.4).on(qubit)
circuit += cirq.ry(params[1]).on(qubit)
ins = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)
outs = tfq.layers.NoisyPQC(circuit, [cirq.Z(qubit)], repetitions=500, sample_based=False)(ins)
vqe = tf.keras.models.Model(inputs=ins, outputs=outs)

N = 30
xs = np.linspace(-np.pi, np.pi, N)
ys = np.linspace(-np.pi, np.pi, N)
zs = np.zeros(shape=(N, N))
inputs = tfq.convert_to_tensor([cirq.Circuit()])
for i in range(len(xs)):
    print(i)
    for j in range(len(ys)):
        vqe.set_weights([np.array([xs[i], ys[j]])])
        zs[i][j] = vqe(inputs).numpy()[0][0]

qubit = cirq.GridQubit(0, 0)
params = sympy.symbols("q0:2")
circuit = cirq.Circuit()
circuit += cirq.ry(params[0]).on(qubit)
circuit += cirq.ry(params[1]).on(qubit)
ins = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)
outs = tfq.layers.PQC(circuit, [cirq.Z(qubit)])(ins)
vqe = tf.keras.models.Model(inputs=ins, outputs=outs)

xs = np.linspace(-np.pi, np.pi, N)
ys = np.linspace(-np.pi, np.pi, N)
zs1 = np.zeros(shape=(N, N))
inputs = tfq.convert_to_tensor([cirq.Circuit()])
for i in range(len(xs)):
    print(i)
    for j in range(len(ys)):
        vqe.set_weights([np.array([xs[i], ys[j]])])
        zs1[i][j] = vqe(inputs).numpy()[0][0]

fig, axs = plt.subplots(1, 2)
axs[1].contourf(xs, ys, zs, 30, cmap='RdGy')
axs[1].set_title("Noisy")
axs[0].contourf(xs, ys, zs1, 30, cmap='RdGy')
axs[0].set_title("Noiseless")
plt.show()
