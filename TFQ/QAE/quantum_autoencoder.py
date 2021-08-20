import tensorflow_quantum as tfq
import cirq 
import sympy
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from itertools import product

def layer(qs, params):
    circ = cirq.Circuit()
    for i in reversed(range(len(qs)-1)):
        circ += cirq.CNOT(qs[i], qs[i+1])
    for i in range(len(qs)):
        circ += cirq.ry(params[2*i]).on(qs[i])
        circ += cirq.rz(params[2*i + 1]).on(qs[i])
    return circ

def make_circuit(qs, state, latent, params, depth, swap_qubit, reference_qubits):
    c = cirq.Circuit()
    enc_params = params[:len(params) // 2]
    dec_params = params[len(params) // 2:]
    for i in range(depth):
        c += layer(qs[:state], enc_params[2 * i * state:2 * (i + 1) * state])
    for i in range(depth):
        c += layer(qs[state - latent:], dec_params[2 * i * state:2 * (i + 1) * state])
    # SWAP Test
    c += cirq.H(swap_qubit)
    for i, j in product(range(state), range(state - latent, len(qs))):
        #c += cirq.CSWAP(swap_qubit, reference_qubits[i], qs[j])
        c += cirq.ControlledGate(sub_gate=cirq.SWAP, num_controls=1).on(swap_qubit, reference_qubits[i], qs[j])
    c += cirq.H(swap_qubit)
    return c


state_qubits = 4
latent_qubits = 1
total_qubits = state_qubits + (state_qubits - latent_qubits)

qubits = [cirq.GridQubit(0, i) for i in range(total_qubits + 1 + state_qubits)]
print(len(qubits))
#states, _ = tfq.datasets.excited_cluster_states(qubits[:state_qubits])
#reference_states, _ = tfq.datasets.excited_cluster_states(qubits[total_qubits + 1:])
#states, _, _, _ = tfq.datasets.xxz_chain(qubits[:state_qubits])
#reference_states, _, _, _ = tfq.datasets.xxz_chain(qubits[total_qubits + 1:])
states, _, _, _ = tfq.datasets.tfi_chain(qubits[:state_qubits])
reference_states, _, _, _ = tfq.datasets.tfi_chain(qubits[total_qubits + 1:])
states = list(states)
reference_states = list(reference_states)
temp = list(zip(states, reference_states))
random.shuffle(temp)
states, reference_states = zip(*temp)

layers = 4

num_params = 4 * state_qubits * layers
parameters = sympy.symbols("q0:%d"%num_params)

train_size = 9 * len(states) // 10
test_size = len(states) - train_size
training_states = states[:train_size]
testing_states = states[train_size:]
reference_states_train = reference_states[:train_size]
reference_states_test = reference_states[train_size:]
train_circuits = [training_states[i]  + reference_states_train[i] for i in range(train_size)]
test_circuits = [testing_states[i] + reference_states_test[i] for i in range(test_size)]
#print(make_circuit(qubits[:total_qubits], state_qubits, latent_qubits, parameters, layers, qubits[total_qubits], qubits[total_qubits + 1:]))
#print(training_states[0]  + reference_states[0] + make_circuit(qubits[:total_qubits], state_qubits, latent_qubits, parameters, layers, qubits[total_qubits], qubits[total_qubits + 1:]))
#print(train_circuits[0])

c = make_circuit(qubits[:total_qubits], state_qubits, latent_qubits, parameters, layers, qubits[total_qubits], qubits[total_qubits + 1:])
readout_operators = [cirq.Z(qubits[total_qubits])]
inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
layer1 = tfq.layers.PQC(c, readout_operators, differentiator=tfq.differentiators.Adjoint())(inputs)
autoencoder = tf.keras.models.Model(inputs=inputs, outputs=layer1)
autoencoder.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(lr=0.1))
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

X_train = tfq.convert_to_tensor(train_circuits)
X_test = tfq.convert_to_tensor(test_circuits)

y_train = np.ones(shape=len(train_circuits))
y_test = np.ones(shape=len(test_circuits))

history = autoencoder.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_test, y_test), callbacks=[callback])

plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Test')
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("1 - Fidelity")
plt.show()
plt.savefig("loss_comp")



