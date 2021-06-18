import tensorflow_quantum as tfq
import cirq 
import sympy
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def create_target_U(num_q):
    cir = cirq.Circuit()
    qubits = [cirq.GridQubit(0, i) for i in range(num_q)]
    for i in range(num_q):
        cir += cirq.H(qubits[i])
        cir += cirq.rz(np.random.uniform(0, 2*np.pi)).on(qubits[i])
    for i in range(num_q - 1):
        cir += cirq.CNOT(qubits[i], qubits[i + 1])
    for i in range(num_q):
        cir += cirq.rz(np.random.uniform(0, 2*np.pi)).on(qubits[i])
    return cir

def ansatz(num_q, qubits):
    cir = cirq.Circuit()
    num_params = num_q * 2
    params = sympy.symbols("params0:%d"%num_params)
    for i in range(num_q):
        cir += cirq.H(qubits[i])
        cir += cirq.rz(params[i]).on(qubits[i])
    for i in range(num_q - 1):
        cir += cirq.CNOT(qubits[i], qubits[i + 1])
    for i in range(num_q):
        cir += cirq.rz(params[i + num_q]).on(qubits[i])
    return cir, params

num_q = 4
qs = [cirq.GridQubit(0, i) for i in range(num_q)]
ans, param = ansatz(num_q, qs)
target_u = create_target_U(num_q)

opt = tf.keras.optimizers.Adam(lr=0.1)

init = tf.Variable(initial_value=np.random.uniform(0, 2 * np.pi, (len(param))), dtype="complex64", trainable=True)
target = tfq.convert_to_tensor([[target_u]])
guess = tfq.convert_to_tensor([ans])

iterations = 50
costs = []
for i in range(iterations):
    with tf.GradientTape() as tape:
        tape.watch(init)
        values = tf.convert_to_tensor(init)
        values = tf.reshape(values, [1, len(values)])
        fidelity = tfq.math.fidelity(guess, tf.convert_to_tensor([s.name for s in param]), values, target)
        cost = 1 - fidelity
    
    print(i, cost.numpy()[0][0])
    grads = tape.gradient(cost, init)
    opt.apply_gradients(zip([grads], [init]))
    costs.append(cost.numpy()[0][0])


plt.plot(costs)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()
plt.savefig("cost_4")
