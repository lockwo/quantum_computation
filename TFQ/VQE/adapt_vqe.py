import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np
import matplotlib.pyplot as plt
from math import prod
import random
import openfermion as of
from openfermionpyscf import generate_molecular_hamiltonian
from scipy.sparse import linalg

def layer(circuit, qubits, parameters):
    for i in range(len(qubits)):
        circuit += cirq.ry(parameters[3*i]).on(qubits[i])
        circuit += cirq.rz(parameters[3*i+1]).on(qubits[i])
        circuit += cirq.ry(parameters[3*i+2]).on(qubits[i])
    for i in range(len(qubits)-1):
        circuit += cirq.CNOT(qubits[i], qubits[i+1])
    circuit += cirq.CNOT(qubits[-1], qubits[0])
    return circuit

def ansatz(circuit, qubits, layers, parameters):
    for i in range(layers):
        params = parameters[3 * i * len(qubits):3 * (i + 1) * len(qubits)]
        circuit = layer(circuit, qubits, params)
    return circuit

def make_vqe(qubits, layers, hamiltonian):
    num_params = layers * 3 * len(qubits)
    params = sympy.symbols('vqe0:%d'%num_params)
    c = ansatz(cirq.Circuit(), qubits, layers, params)
    ins = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)
    pqc = tfq.layers.PQC(c, hamiltonian, differentiator=tfq.differentiators.Adjoint())(ins)
    vqe = tf.keras.models.Model(inputs=ins, outputs=pqc)
    return vqe

def optimize_vqe_gradient(vqe, init, track=[]):
    old = np.inf
    inputs = tfq.convert_to_tensor([cirq.Circuit()])
    counter = 0
    vqe.set_weights([init])
    opt = tf.keras.optimizers.Adam(learning_rate=0.1)
    energy = 0
    while counter < 2000:
        with tf.GradientTape() as tape:
            guess = vqe(inputs)
        grads = tape.gradient(guess, vqe.trainable_variables)
        opt.apply_gradients(zip(grads, vqe.trainable_variables))
        guess = guess.numpy()[0][0]
        energy = guess
        if abs(guess - old) < 1e-5 and counter > 100:
            break
        old = guess
        counter += 1
        track.append(guess)

    return energy, track

def generate_circuits(base_cir, ops, p, qubits):
    circuit = []
    single_qubit_ops = ops[0]
    two_qubit_ops = ops[1]
    for op in single_qubit_ops:
        for q in qubits:
            circuit.append(base_cir + op(p).on(q))
    for op in two_qubit_ops:
        for i in range(len(qubits) - 1):
            circuit.append(base_cir + op.on(qubits[i], qubits[i + 1])**p)
        circuit.append(base_cir + op.on(qubits[-1], qubits[0])**p)
    return circuit

def adapt_vqe(h, op_pool, qubits):
    expectation_layer = tfq.layers.Expectation()
    adapt_iter = 0
    params = []
    symbols = []
    base_circuit = cirq.Circuit()
    for q in qubits:
        base_circuit += cirq.I(q)
    opt = tf.keras.optimizers.Adam(learning_rate=0.1)
    track = []
    adapt_prev_iter = np.inf
    while adapt_iter < 18:
        params.append(np.random.uniform(0, np.pi))
        symbols.append(sympy.symbols("a%d"%adapt_iter))
        circuits = generate_circuits(base_circuit, op_pool, symbols[-1], qubits)
        var = tf.Variable(params, dtype=tf.float32, trainable=True)
        grads = []
        for c in circuits:
            with tf.GradientTape() as tape:
                tape.watch(var)
                exp_val = expectation_layer(c, symbol_names=[s.name for s in symbols], symbol_values=[var], operators=h)
            grads.append(tf.math.abs(tape.gradient(exp_val, var))[-1])


        grads_ = [i for i in grads if i > 1e-3]
        if len(grads_) == 0:# or adapt_iter < 6:
            base_circuit = random.choice(circuits)
        else:
            base_circuit = circuits[np.argmax(grads)]

        counter = 0
        old = np.inf
        guess = np.inf
        while counter < 2000:
            with tf.GradientTape() as tape:
                tape.watch(var)
                guess = expectation_layer(base_circuit, symbol_names=[s.name for s in symbols], symbol_values=[var], operators=h)
            grads = tape.gradient(guess, var)
            opt.apply_gradients(zip([grads], [var]))
            guess = guess.numpy()[0][0]
            energy = guess
            track.append(energy)
            if abs(guess - old) < 1e-4 and counter > 20:
                break
            old = guess
            counter += 1
        if abs(adapt_prev_iter - track[-1]) < 1e-4: # and adapt_iter > 10
            break
        adapt_prev_iter = track[-1]
        params = var.numpy().tolist()
        adapt_iter += 1
        print(adapt_iter, guess)
        print(base_circuit)
    return track

# VQE Hyperparameters
layers = 3

basis = 'sto-3g'
multiplicity = 1
charge = 0
dists = [0.2, 0.8, 2.0, 3.0]
for dist in dists:
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., dist))]
    molecular_hamiltonian = generate_molecular_hamiltonian(geometry, basis, multiplicity, charge)
    q = of.count_qubits(molecular_hamiltonian)
    qs = [cirq.GridQubit(0, i) for i in range(q)]
    jw_operator = of.transforms.jordan_wigner(molecular_hamiltonian)
    hamiltonian_jw_sparse = of.get_sparse_operator(jw_operator)
    eigs, _ = linalg.eigsh(hamiltonian_jw_sparse, k=1, which='SA')
    hamiltonian = of.transforms.qubit_operator_to_pauli_sum(jw_operator, qubits=qs)

    operator_pool = [[cirq.rx, cirq.ry, cirq.rz], [cirq.CNOT]]

    adapt_track = adapt_vqe(hamiltonian, operator_pool, qs)

    vqe = make_vqe(qs, layers, hamiltonian)
    initial_value = tf.random.uniform(shape=[layers * 3 * q], minval=0, maxval=2 * np.pi)
    _, grad_vqe = optimize_vqe_gradient(vqe, initial_value)

    plt.title(str(dist))
    plt.plot(grad_vqe, label='Gradient VQE')
    plt.plot(adapt_track, label='ADAPT VQE')
    plt.axhline(y=eigs[0], color="tab:red", ls="--", label="Target")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Energy")
    plt.show()
