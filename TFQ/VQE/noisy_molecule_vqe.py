import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import openfermion as of
from openfermionpyscf import generate_molecular_hamiltonian
from scipy.sparse import linalg
import numpy as np
import matplotlib.pyplot as plt

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

def make_vqe_noisy(qubits, layers, hamiltonian):
    num_params = layers * 3 * len(qubits)
    params = sympy.symbols('vqe0:%d'%num_params)
    c = ansatz(cirq.Circuit(), qubits, layers, params)
    c = c.with_noise(cirq.depolarize(p=0.01))
    ins = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)
    pqc = tfq.layers.NoisyPQC(c, hamiltonian, repetitions=1000, sample_based=True, differentiator=tfq.differentiators.ParameterShift())(ins)
    vqe = tf.keras.models.Model(inputs=ins, outputs=pqc)
    return vqe

def optimize_vqe_gradient(vqe, init):
    old = np.inf
    inputs = tfq.convert_to_tensor([cirq.Circuit()])
    counter = 0
    vqe.set_weights([init])
    opt = tf.keras.optimizers.Adam(learning_rate=0.1) # Empirically justified (Lockwood, 2022)
    energy = 0
    while counter < 200:
        with tf.GradientTape() as tape:
            guess = vqe(inputs)
        grads = tape.gradient(guess, vqe.trainable_variables)
        opt.apply_gradients(zip(grads, vqe.trainable_variables))
        guess = guess.numpy()[0][0]
        energy = guess
        if abs(guess - old) < 1e-5:
            break
        old = guess
        counter += 1

    return energy

diatomic_bond_length = 0.2
interval = 0.1
max_bond_length = 4.0 
basis = 'sto-3g'
multiplicity = 1
charge = 0
ground_energies_real = []
ground_energies_vqe = []
ground_energies_vqe_noisy = []
bond_lengths = []
k = 2

# VQE Hyperparameters
layers = 2

while diatomic_bond_length <= max_bond_length:
    print(diatomic_bond_length, max_bond_length)
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., diatomic_bond_length))]
    molecular_hamiltonian = generate_molecular_hamiltonian(geometry, basis, multiplicity, charge)
    n_qubits = of.count_qubits(molecular_hamiltonian)
    qs = [cirq.GridQubit(0, i) for i in range(n_qubits)]
    jw_operator = of.transforms.jordan_wigner(molecular_hamiltonian)
    hamiltonian_jw_sparse = of.get_sparse_operator(jw_operator)
    eigs, _ = linalg.eigsh(hamiltonian_jw_sparse, k=k, which='SA')
    hamiltonian = of.transforms.qubit_operator_to_pauli_sum(jw_operator, qubits=qs)

    vqe = make_vqe(qs, layers, hamiltonian)
    vqe_noisy = make_vqe_noisy(qs, layers, hamiltonian)
    initial_value = tf.random.uniform(shape=[layers * 3 * n_qubits], minval=0, maxval=2 * np.pi)
    ground_gradient = optimize_vqe_gradient(vqe, initial_value)
    ground_gradient_noisy = optimize_vqe_gradient(vqe_noisy, initial_value)
    ground_energies_vqe.append(ground_gradient)
    ground_energies_vqe_noisy.append(ground_gradient_noisy)
    ground_energies_real.append(eigs[0])
    bond_lengths.append(diatomic_bond_length)

    diatomic_bond_length += interval


plt.scatter(bond_lengths, ground_energies_vqe, label='VQE Predicted Ground State', marker='o', facecolors="None", edgecolor='blue')
plt.scatter(bond_lengths, ground_energies_vqe_noisy, label='Noisy VQE Predicted Ground State', marker='s', facecolors="None", edgecolor='red')
plt.plot(bond_lengths, ground_energies_real, label='Ground State', color='blue')
plt.xlabel("Interatomic Distance (Angstroms)")
plt.ylabel("Energy (Hartree)")
plt.legend()
plt.show()
