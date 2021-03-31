import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np
import matplotlib.pyplot as plt

# https://www.tensorflow.org/quantum/tutorials/barren_plateaus#2_generating_random_circuits
def generate_circuit(qubits, depth, param):
    circuit = cirq.Circuit()
    for qubit in qubits:
        circuit += cirq.ry(np.pi / 4.0)(qubit)

    for d in range(depth):
        for i, qubit in enumerate(qubits):
            random_n = np.random.uniform()
            random_rot = np.random.uniform(0, 2.0 * np.pi) if i != 0 or d != 0 else param
            if random_n > 2. / 3.:
                circuit += cirq.rz(random_rot)(qubit)
            elif random_n > 1. / 3.:
                circuit += cirq.ry(random_rot)(qubit)
            else:
                circuit += cirq.rx(random_rot)(qubit)

        for src, dest in zip(qubits, qubits[1:]):
            circuit += cirq.CZ(src, dest)

    return circuit

def grad_variance(circuits, qubits, symbol, reps, ops):
    if ops == "all":
        readout_ops = sum([cirq.Z(i) for i in qubits])
    else:
        readout_ops = [cirq.Z(qubits[0]) * cirq.Z(qubits[1])]
    rep = reps
    diff = tfq.differentiators.ParameterShift()

    expectation = tfq.layers.SampledExpectation(differentiator=diff)
    circuit_tensor = tfq.convert_to_tensor(circuits)
    values_tensor = tf.convert_to_tensor(np.random.uniform(0, 2 * np.pi, (len(circuits), 1)).astype(np.float32))

    with tf.GradientTape() as tape:
        tape.watch(values_tensor)
        forward = expectation(circuit_tensor, operators=readout_ops, repetitions=rep, symbol_names=[symbol], symbol_values=values_tensor)

    grads = tape.gradient(forward, values_tensor)
    grad_var = tf.math.reduce_std(grads, axis=0)
    return grad_var.numpy()[0]

def q_loop(range_q, depth, n_cir, reps, op):
    varss = []
    for i in range(2, range_q//2):
        i = 2 * i
        print(i, range_q)
        qubits = [cirq.GridQubit(0, j) for j in range(i)]
        symbol = sympy.symbols("param")
        circuits = [generate_circuit(qubits, depth, symbol) for _ in range(n_cir)]
        varss.append(grad_variance(circuits, qubits, symbol, reps, op))
    return varss

def d_loop(q, range_d, n_cir, reps, op):
    varss = []
    for i in range(1, range_d//20):
        i = 20 * i
        print(i, range_d)
        qubits = [cirq.GridQubit(0, j) for j in range(q)]
        symbol = sympy.symbols("param")
        circuits = [generate_circuit(qubits, i, symbol) for _ in range(n_cir)]
        varss.append(grad_variance(circuits, qubits, symbol, reps, op))
    return varss

n_cir = 100
qs = 16
d = 100
results_all = q_loop(qs, d, n_cir, 1000, "all")
results_paper = q_loop(qs, d, n_cir, 1000, "one")
xs = [i * 2 for i in range(2, qs//2)]
plt.plot(xs, results_all, label="sum(Z) Measure")
plt.plot(xs, results_paper, label="ZZ Measure")
plt.xlabel("Number of Qubits")
plt.ylabel("Variance of Gradients")
plt.legend()
plt.show()

ds = 160
n_cir = 100
op = "one"
results_12 = d_loop(12, ds, n_cir, 1000, op)
results_6 = d_loop(6, ds, n_cir, 1000, op)
xs = [20 * i for i in range(1, ds//20)]
plt.plot(xs, results_12, label='12 Qubits')
plt.plot(xs, results_6, label='6 Qubits')
plt.xlabel("Depth")
plt.ylabel("Variance of Gradients")
plt.legend()
plt.show()
