import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np
import random
import matplotlib.pyplot as plt
from copy import deepcopy
from functools import reduce 
import operator

def layer(circuit, qubits, parameters):
    for i in range(len(qubits)):
        circuit.append([cirq.rx(parameters[2*i]).on(qubits[i])])
        circuit.append([cirq.rz(parameters[2*i+1]).on(qubits[i])])
    for i in range(len(qubits)-1):
        circuit.append([cirq.CNOT(qubits[i], qubits[i+1])])
    return circuit

def ansatz(circuit, qubits, layers, parameters):
    for i in range(layers):
        p = parameters[2 * i * len(qubits):2 * (i + 1) * len(qubits)]
        circuit = layer(circuit, qubits, p)
    return circuit

def hamiltonian(circuit, qubits, ham):
    for i in range(len(qubits)):
        if ham[i] == "x":
            circuit.append(cirq.ry(-np.pi/2).on(qubits[i]))
        elif ham[i] == "y":
            circuit.append(cirq.rx(np.pi/2).on(qubits[i]))
    return circuit

def create_vqe(qubits, layers, parameters, ham):
    circuit = ansatz(cirq.Circuit(), qubits, layers, parameters)
    circuit = hamiltonian(circuit, qubits, ham)
    return circuit

def prod(iterable):
    return reduce(operator.mul, iterable, 1)

def expcost(qubits, ham):
    return prod([cirq.Z(qubits[i]) for i in range(len(qubits)) if ham[i] != "i"])

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
    return sorted(eig)[0]

possibilities = ["i", "x", "y", "z"]
l = 4
q = 3

hamilton = [[random.choice(possibilities) for _ in range(q)] for _ in range(l)]
h_weights = [random.uniform(-1, 1) for _ in range(l)]

hamilton = [["x", "i", "i"], ["i", "x", "i"], ["i", "z", "z"], ["i", "i", "x"]]
h_weights = [0.4977616234240615, 0.5635396435844906, 0.32588875859719557, 0.18913602999217294]
lay = 2

qubits = [cirq.GridQubit(0, i) for i in range(q)]
num_params = lay * 2 * q
params = sympy.symbols('vqe0:%d'%num_params)

class VQE(tf.keras.layers.Layer):
    def __init__(self, num_weights, circuits, ops) -> None:
        super(VQE, self).__init__()
        self.w = tf.Variable(np.random.uniform(0, np.pi, (1, num_weights)), dtype=tf.float32)
        self.layers = [tfq.layers.ControlledPQC(circuits[i], ops[i], repetitions=1000, differentiator=tfq.differentiators.ParameterShift()) \
            for i in range(len(circuits))]

    def call(self, input):
        return sum([self.layers[i]([input, self.w]) for i in range(len(self.layers))])

ins = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)
inputs = tfq.convert_to_tensor([cirq.Circuit()])
vqe_components = []
print(hamilton, h_weights)
cs = []
op = []

for i in range(len(hamilton)):
    readout_ops = h_weights[i] * expcost(qubits, hamilton[i])
    op.append(readout_ops)
    #print(create_vqe(qubits, lay, params, hamilton[i]), readout_ops)
    cs.append(create_vqe(qubits, lay, params, hamilton[i]))

v = VQE(num_params, cs, op)(ins)
vqe_model = tf.keras.models.Model(inputs=ins, outputs=v)
vqe_model.summary()
print(vqe_model.trainable_variables)

r = real_min(deepcopy(hamilton), h_weights)

#opt = tf.keras.optimizers.Adam(lr=0.01)
opt = tf.keras.optimizers.Adagrad(lr=0.2)
N = 100

tfq_i = []

for i in range(N):
    with tf.GradientTape() as tape:
        guess = vqe_model(inputs)
    grads = tape.gradient(guess, vqe_model.trainable_variables)
    opt.apply_gradients(zip(grads, vqe_model.trainable_variables))
    guess = guess.numpy()
    tfq_i.append(guess[0][0])
    if i % 20 == 0:
        print("Epoch {}/{}, Guess {}, Real {}".format(i, N, guess[0][0], r))

plt.plot(tfq_i, label="TFQ")
plt.plot([r for _ in range(len(tfq_i))], label="Real")
plt.legend()
plt.ylabel("Eigenvalue")
plt.xlabel("Iterations")
plt.show()
