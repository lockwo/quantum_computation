import cirq
import sympy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.optimize import minimize
import random

'''
The Swap test is a simple quantum circuit which, given two states, allows you to compute how much do they differ from each other.

Provide a variational (also called parametric) circuit which is able to generate the most general 1 qubit state. 
By most general 1 qubit state we mean that there exists a set of the parameters in the circuit such that any point in the Bloch sphere can be reached. 
Check that the circuit works correctly by showing that by varying randomly the parameters of your circuit you can reproduce correctly the Bloch sphere.

Use the circuit built in step 1) and, using the SWAP test, find the best choice of your parameters to reproduce a randomly generated quantum state made with 1 qubit.

Suppose you are given with a random state, made by N qubits, for which you only know that it is a product state and each of the qubits are in the state | 0 > or | 1>. 
By product state we mean that it can be written as the product of single qubit states, without the need to do any summation. For example, the state
|a> = |01>
Is a product state, while the state
|b> = |00> + |11>
Is not.

Perform a qubit by qubit SWAP test to reconstruct the state. This part of the problem can be solved via a simple grid search.
'''

qubit = cirq.GridQubit(0, 0)
sim = cirq.Simulator()

# Problem 1

results = []
for i in range(10000):
    circuit = cirq.Circuit()
    params = np.random.uniform(0, 2*np.pi, 3)
    circuit.append(cirq.rz(params[0]).on(qubit))
    circuit.append(cirq.ry(params[1]).on(qubit))
    circuit.append(cirq.rz(params[2]).on(qubit))
    final = sim.simulate(circuit)
    results.append(final.bloch_vector_of(qubit))

results = np.asarray(results)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(results[:,0], results[:,1], results[:,2], s=5)
plt.show()

def bitstring(bits):
    return ''.join(str(int(b)) for b in bits)

def SWAP_test(circuit, qubits):
    test_qubit = cirq.GridQubit(1, 0)
    circuit.append(cirq.H(test_qubit))
    circuit.append(cirq.SWAP(qubits[0], qubits[1]).controlled_by(test_qubit))
    circuit.append(cirq.H(test_qubit))
    circuit.append(cirq.measure(test_qubit, key='result'))
    m = 1000
    result = sim.run(circuit, repetitions=m)
    frequencies = result.histogram(key='result', fold_func=bitstring)
    prob = frequencies['0'] / m 
    return prob

# Problem 2

qubit2 = cirq.GridQubit(0, 1)
circuit = cirq.Circuit()
params = np.random.uniform(0, 2*np.pi, 3)
circuit.append(cirq.rz(params[0]).on(qubit))
circuit.append(cirq.ry(params[1]).on(qubit))
circuit.append(cirq.rz(params[2]).on(qubit))

params2 = sympy.symbols('a b c')
circuit.append(cirq.rz(params2[0]).on(qubit2))
circuit.append(cirq.ry(params2[1]).on(qubit2))
circuit.append(cirq.rz(params2[2]).on(qubit2))

maxx = 0
maxx_params = None
print("Original Parameters:", params)
print(circuit)
def f(x):
    resolver = cirq.ParamResolver({'a': x[0], 'b': x[1], 'c': x[2]})
    resolved_circuit = cirq.resolve_parameters(circuit, resolver)
    prob = SWAP_test(resolved_circuit, [qubit, qubit2])
    return 1 - prob

opt = minimize(f, x0=np.random.uniform(0, 2*np.pi, 3), method='Powell')
maxx_params = opt['x']
maxx = 1 - opt['fun']

old_circuit = cirq.Circuit()
old_circuit.append(cirq.rz(params[0]).on(qubit))
old_circuit.append(cirq.ry(params[1]).on(qubit))
old_circuit.append(cirq.rz(params[2]).on(qubit))

print("Recovered Parameters:", maxx_params, "with confidence:", maxx)
new_circuit = cirq.Circuit()
new_circuit.append(cirq.rz(maxx_params[0]).on(qubit2))
new_circuit.append(cirq.ry(maxx_params[1]).on(qubit2))
new_circuit.append(cirq.rz(maxx_params[2]).on(qubit2))
t1 = sim.simulate(old_circuit).bloch_vector_of(qubit)
t2 = sim.simulate(new_circuit).bloch_vector_of(qubit2)
print("Real statevector", t1, "predicted statevector:", t2)
print("Off by:", t1 - t2)

# Problem 3

def get_random_state(N):
    state = []
    for _ in range(N):
        if random.random() < 0.5:
            state.append(1)
        else:
            state.append(0)
    return state

def generate_state(state, qubits):
    circuit = cirq.Circuit()
    for i in range(len(state)):
        if state[i] == 1:
            circuit.append(cirq.X(qubits[i]))
        else:
            circuit.append(cirq.I(qubits[i]))
    return circuit

num = 10
qubits = [cirq.GridQubit(0, i) for i in range(num)]
state = get_random_state(num)

print("Actual state:")
print(state)

pred = []
for i in range(num):
    cir = generate_state(state, qubits)
    extra = cirq.GridQubit(0, num)
    cir.append(cirq.X(extra))
    swap = SWAP_test(cir, [extra, qubits[i]])
    if swap < 0.75:
        pred.append(0)
    else:
        pred.append(1)

print("Predicted:")
print(pred)
