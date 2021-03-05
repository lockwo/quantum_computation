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
        circuit.append([cirq.ry(parameters[3*i]).on(qubits[i])])
        circuit.append([cirq.rz(parameters[3*i+1]).on(qubits[i])])
        circuit.append([cirq.ry(parameters[3*i+2]).on(qubits[i])])
    for i in range(len(qubits)-1):
        circuit.append([cirq.CNOT(qubits[i], qubits[i+1])])
    circuit.append([cirq.CNOT(qubits[-1], qubits[0])])
    return circuit

def ansatz(circuit, qubits, layers, parameters):
    for i in range(layers):
        p = parameters[3 * i * len(qubits):3 * (i + 1) * len(qubits)]
        circuit = layer(circuit, qubits, p)
    return circuit

def hamiltonian(circuit, qubits, ham):
    for i in range(len(qubits)):
        if ham[i] == "x":
            circuit.append(cirq.ry(-np.pi/2).on(qubits[i]))
        elif ham[i] == "y":
            circuit.append(cirq.rx(np.pi/2).on(qubits[i]))
    return circuit

def create_vqe(init, qubits, layers, parameters, ham):
    circuit = ansatz(init, qubits, layers, parameters)
    circuit = hamiltonian(circuit, qubits, ham)
    return circuit

def prod(iterable):
    return reduce(operator.mul, iterable, 1)

def expcost(qubits, ham):
    return prod([cirq.Z(qubits[i]) for i in range(len(qubits)) if ham[i] != "i"])

def real_min(h, w, k):
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
    if k == "all":
        return sorted(eig)
    return sorted(eig)[:k]

class VQE_sub(tf.keras.layers.Layer):
    def __init__(self, circuits, ops) -> None:
        super(VQE_sub, self).__init__()
        self.layers = [tfq.layers.ControlledPQC(circuits[i], ops[i], repetitions=1000, differentiator=tfq.differentiators.ParameterShift()) \
        #self.layers = [tfq.layers.ControlledPQC(circuits[i], ops[i], differentiator=tfq.differentiators.ParameterShift()) \
            for i in range(len(circuits))]

    def call(self, inputs):
        return sum([self.layers[i]([inputs[0], inputs[1]]) for i in range(len(self.layers))])

class SSVQE(tf.keras.layers.Layer):
    def __init__(self, num_weights, circuits, ops, k) -> None:
        super().__init__()
        self.w = tf.Variable(np.random.uniform(0, np.pi, (1, num_weights)), dtype=tf.float32)
        self.hams = []
        self.k = k
        for i in range(k):
            self.hams.append(VQE_sub(circuits[i], ops[i]))

    def call(self, inputs):
        total = 0
        calls = []
        for i in range(self.k):
            c = self.hams[i]([inputs, self.w])
            calls.append(c)
            if i == 0:
                total += c
            else:
                total += ((0.65 - ((i - 1) * 0.1)) * c)
        return total, calls

possibilities = ["i", "x", "y", "z"]

l = 5
lay = 3
q = 3

hamilton = [[random.choice(possibilities) for _ in range(q)] for _ in range(l)]
h_weights = [random.uniform(-1, 1) for _ in range(l)]

#h_weights = [0.35807927646889326, 0.7556205249987815, 0.04828309125493235, 0.07927207111541623]
#hamilton = [["x", "i", "i"], ["i", "x", "i"], ["i", "i", "x"], ["i", "z", "z"]]

hamilton = [["x", "i", "i"], ["i", "x", "i"], ["i", "z", "z"], ["i", "i", "x"]]
h_weights = [0.4977616234240615, 0.5635396435844906, 0.32588875859719557, 0.18913602999217294]

qubits = [cirq.GridQubit(0, i) for i in range(q)]
num_params = lay * 3 * q
params = sympy.symbols('vqe0:%d'%num_params)

ins = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)
inputs = tfq.convert_to_tensor([cirq.Circuit()])
vqe_components = []
print(hamilton, h_weights)
cs = []
op = []
k = 3

for j in range(k):
    cs.append([])
    op.append([])
    for i in range(len(hamilton)):
        readout_ops = h_weights[i] * expcost(qubits, hamilton[i])
        op[j].append(readout_ops)
        if j == 0:
            cs[j].append(create_vqe(cirq.Circuit(), qubits, lay, params, hamilton[i]))
        else:
            #cs[j].append(create_vqe(cirq.Circuit(cirq.X(qubits[j % 2])), qubits, lay, params, hamilton[i]))
            cs[j].append(create_vqe(cirq.Circuit(cirq.X(qubits[j-1])), qubits, lay, params, hamilton[i]))

v = SSVQE(num_params, cs, op, k)(ins)
vqe_model = tf.keras.models.Model(inputs=ins, outputs=v[0])
vqe_model.summary()
print(vqe_model.trainable_variables)

r = real_min(deepcopy(hamilton), h_weights, k)

opt = tf.keras.optimizers.Adam(lr=0.1)
#opt = tf.keras.optimizers.Adagrad(lr=0.2)
N = 140

tfq_0 = []
tfq_1 = []
tfq_2 = []

subs = SSVQE(num_params, cs, op, k)

for i in range(N):
    subs.w = vqe_model.trainable_variables[0]
    with tf.GradientTape() as tape:
        guess = vqe_model(inputs)
    grads = tape.gradient(guess, vqe_model.trainable_variables)
    opt.apply_gradients(zip(grads, vqe_model.trainable_variables))
    guess = guess.numpy()
    es = [i.numpy()[0][0] for i in subs(inputs)[1]]
    tfq_0.append(es[0])
    tfq_1.append(es[1])
    tfq_2.append(es[2])
    if i % 20 == 0:
        print("Epoch {}/{}, Loss {}, E's {}, Real{}".format(i, N, guess[0][0], es, r))

plt.plot(tfq_0, label="TFQ")
plt.plot(tfq_1, label="TFQ")
plt.plot(tfq_2, label="TFQ")
for i in r:
    plt.plot([i for _ in range(len(tfq_0))], label="Real")
plt.legend()
plt.ylabel("Energy")
plt.xlabel("Iterations")
plt.show()
