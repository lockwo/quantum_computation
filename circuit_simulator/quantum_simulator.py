import numpy as np 
from gates import *
import math 
from collections import defaultdict 

'''
Learning by doing: the best way to understand the basics of quantum computation is to implement a quantum circuit simulator. 
This task is suitable both for people from computer sciences who want to learn about quantum computing, and for people from math/physics who want to exercise coding.

Detailed description of the task with some learning resources and examples can be found in this jupyter notebook

It is expected that simulator can perform following:
initialize state
read program, and for each gate:
calculate matrix operator
apply operator (modify state)
perform multi-shot measurement of all qubits using weighted random technique
'''

def ground_state_qubits(num):
    q = np.zeros(shape=(int(2**num), 1))
    q[0][0] = 1
    return q

def get_op(qubits, gate):
    q = gate.on
    num_q = int(math.log(len(qubits), 2))
    if hasattr(gate, 'two_qubit'):
        control = q[0]
        target = q[1]
        qubit_ops1 = []
        qubit_ops2 = []
        for i in range(num_q):
            if i == control:
                qubit_ops1.append(outer_00)
                qubit_ops2.append(outer_11)
            elif i == target:
                qubit_ops1.append(I().op)
                qubit_ops2.append(gate.op)
            else:
                qubit_ops1.append(I().op)
                qubit_ops2.append(I().op)
        op1 = np.kron(qubit_ops1[0], qubit_ops1[1])
        op2 = np.kron(qubit_ops2[0], qubit_ops2[1])
        for i in range(2, num_q):
            op1 = np.kron(op1, qubit_ops1[i])
            op2 = np.kron(op2, qubit_ops2[i])
        op = op1 + op2
    else:
        if num_q == 1:
            return gate.op
        qubit_ops = []
        for i in range(num_q):
            if i == q:
                qubit_ops.append(gate.op)
            else:
                qubit_ops.append(I().op)
        op = np.kron(qubit_ops[0], qubit_ops[1])
        for i in range(2, num_q):
            op = np.kron(op, qubit_ops[i])
    return op

def default_val():
    return 0

def measure(qubits, num):
    qubits = qubits.flatten()
    qubits = np.real(qubits * np.conj(qubits))
    measurements = np.random.choice(len(qubits), num, p=qubits)
    m = defaultdict(default_val)
    n = int(math.log(len(qubits), 2))
    for i in measurements:
        k = bin(i)[2:].zfill(n)
        if k in m:
            m[k] += 1
        else:
            m[k] = 1
    return m

def run_circuit(qubits, circuit, measurements):
    for i in range(len(circuit)):
        operation = get_op(qubits, circuit[i])
        qubits = operation @ qubits
    return measure(qubits, measurements)

qubits = ground_state_qubits(1)
x_test = [
    X(0)
]
m = run_circuit(qubits, x_test, 1000)
print("Singe X Test:", dict(m))

qubits = ground_state_qubits(2)
x_test2 = [
    X(1)
]
m = run_circuit(qubits, x_test2, 1000)
print("Two qubit X test:", dict(m))

qubits = ground_state_qubits(2)
bell_state = [
    H(0),
    Controlled(X(), [0, 1])
]
m = run_circuit(qubits, bell_state, 1000)
print("Bell state test:", dict(m))

qubits = ground_state_qubits(10)
ghz_state = [
    H(0),
    Controlled(X(), [0, 1]),
    Controlled(X(), [1, 2]),
    Controlled(X(), [2, 3]),
    Controlled(X(), [3, 4]),
    Controlled(X(), [4, 5]),
    Controlled(X(), [5, 6]),
    Controlled(X(), [6, 7]),
    Controlled(X(), [7, 8]),
    Controlled(X(), [8, 9])
]
m = run_circuit(qubits, ghz_state, 1000)
print("GHZ state:", dict(m))

qubits = ground_state_qubits(2)
grovers_algorithm = [
    H(0), # Initial Superposition
    H(1),
    Controlled(Z(), [0, 1]), # Oracle for |11>
    H(0), # Diffusion Operator
    H(1),
    Z(0),
    Z(1),
    Controlled(Z(), [0, 1]),
    H(0),
    H(1)
]
m = run_circuit(qubits, grovers_algorithm, 1000)
print("Grovers algorithm for |11>:", dict(m))

# VQE 
def hamiltonian(parameters):
    ham = [0.5, 2.5, 1, 2]
    parameter_gate = Ry(parameters[0], 0)

    x_circuit = [
        parameter_gate, 
        Ry(-np.pi/2, 0),
    ]

    y_circuit = [
        parameter_gate,
        Rx(np.pi/2, 0)
    ]

    z_circuit = [
        parameter_gate
    ]

    measures = 1000

    qubits = ground_state_qubits(1)
    xs = run_circuit(qubits, x_circuit, measures)
    xs = (xs["0"] - xs["1"]) 
    qubits = ground_state_qubits(1)
    ys = run_circuit(qubits, y_circuit, measures)
    ys = (ys["0"] - ys["1"]) 
    qubits = ground_state_qubits(1)
    zs = run_circuit(qubits, z_circuit, measures)
    zs = (zs["0"] - zs["1"]) 

    xs /= measures
    ys /= measures
    zs /= measures

    return ham[0] + ham[1] * xs + ham[2] * ys + ham[3] * zs

import matplotlib.pyplot as plt

N = 100
xs = np.linspace(0, 2*np.pi, N)
energies = []
for i in range(N):
    energies.append(hamiltonian([xs[i]]))

plt.plot(xs, energies)
plt.xlabel("Parameter")
plt.ylabel("Energy/Eigenvalue")
plt.title("VQE")
plt.show()
