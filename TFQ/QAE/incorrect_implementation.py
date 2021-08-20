import tensorflow_quantum as tfq
import cirq 
import sympy
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

def layer(qs, params):
    circ = cirq.Circuit()
    for i in reversed(range(len(qs)-1)):
        circ += cirq.CNOT(qs[i], qs[i+1])
    for i in range(len(qs)):
        circ += cirq.ry(params[2*i]).on(qs[i])
        circ += cirq.rz(params[2*i + 1]).on(qs[i])
    return circ

def make_circuit(qs, state, latent, params, depth):
    c = cirq.Circuit()
    enc_params = params[:len(params) // 2]
    dec_params = params[len(params) // 2:]
    for i in range(depth):
        c += layer(qs[:state], enc_params[2 * i * state:2 * (i + 1) * state])
    for i in range(depth):
        c += layer(qs[state - latent:], dec_params[2 * i * state:2 * (i + 1) * state])
    return c

def mod_states(ss, qus):
    c = cirq.Circuit()
    for i in qus:
        c += cirq.I(i)
    c += ss
    return c

state_qubits = 4
latent_qubits = 1
total_qubits = state_qubits + (state_qubits - latent_qubits)

qubits = [cirq.GridQubit(0, i) for i in range(total_qubits)]
states, _ = tfq.datasets.excited_cluster_states(qubits[:state_qubits])
#states, _, _, _ = tfq.datasets.xxz_chain(qubits[:state_qubits])
#states, _, _, _ = tfq.datasets.tfi_chain(qubits[:state_qubits])
states = [mod_states(i, qubits) for i in states]
random.shuffle(states)

layers = 2

num_params = 4 * state_qubits * layers
parameters = sympy.symbols("q0:%d"%num_params)

train_size = 9 * len(states) // 10
test_size = len(states) - train_size
training_states = states[:train_size]
testing_states = states[train_size:]
#print(states[0])
train_circuits = [training_states[i] + make_circuit(qubits, state_qubits, latent_qubits, parameters, layers) for i in range(train_size)]
test_circuits = [testing_states[i] + make_circuit(qubits, state_qubits, latent_qubits, parameters, layers) for i in range(test_size)]
#print(make_circuit(qubits, state_qubits, latent_qubits, parameters, enc_layers, dec_layers))

opt = tf.keras.optimizers.Adam(lr=0.02)

training_target = tfq.convert_to_tensor(training_states)
testing_target = tfq.convert_to_tensor(testing_states)

init = tf.Variable(initial_value=np.random.uniform(0, 2 * np.pi, (1, num_params)), dtype="complex64", trainable=True)
train_guess = tfq.convert_to_tensor(train_circuits)
test_guess = tfq.convert_to_tensor(test_circuits)
names = tf.convert_to_tensor([s.name for s in parameters])
training_target = tf.expand_dims(training_target, axis=1)
testing_target = tf.expand_dims(testing_target, axis=1)

training_costs = []
testing_costs = []
prev = np.inf
tol = 1e-5
iterr = 1
patience = 0
while True:
    with tf.GradientTape() as tape:
        tape.watch(init)
        values = tf.convert_to_tensor(init)
        train_values = tf.tile(values, [train_size, 1])
        train_fidelity = tfq.math.fidelity(train_guess, names, train_values, training_target)       
        train_cost = 1 - tf.math.reduce_mean(train_fidelity)
        
    values = tf.convert_to_tensor(init)
    test_values = tf.tile(values, [test_size, 1])
    test_fidelity = tfq.math.fidelity(test_guess, names, test_values, testing_target)       
    test_cost = 1 - tf.math.reduce_mean(test_fidelity)

    train_val = train_cost.numpy()
    test_val = test_cost.numpy()
    print(iterr, train_val, test_val)
    grads = tape.gradient(train_cost, init)
    opt.apply_gradients(zip([grads], [init]))
    training_costs.append(train_val)
    testing_costs.append(test_val)
    if prev < train_val:
        patience += 1
    else:
        patience = 0
    if abs(prev - train_val) < tol or patience > 10:
        break
    prev = train_val
    iterr += 1

plt.plot(training_costs, label='Train')
plt.plot(testing_costs, label='Test')
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("1 - Fidelity")
plt.show()
plt.savefig("loss_comp")



