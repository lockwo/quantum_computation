from skopt import gp_minimize
import cirq 
import sympy
import numpy as np
import matplotlib.pyplot as plt


def create_target_X():
    cir = cirq.Circuit()
    qubits = cirq.GridQubit(0, 0)
    cir += cirq.X(qubits)
    return cir

def create_target_H():
    cir = cirq.Circuit()
    qubits = cirq.GridQubit(0, 0)
    cir += cirq.H(qubits)
    return cir

def initial_ansatz(max_depth):
    cir = cirq.Circuit()
    gates = [cirq.rx(np.pi/2), cirq.rz, cirq.I]
    qubits = cirq.GridQubit(0, 0)
    params = sympy.symbols("params0:%d"%max_depth)
    ret_param = []
    iss = 0
    for i in range(max_depth):
        j = np.random.uniform(0, 1)
        if j < 1/3:
            cir += gates[0].on(qubits)
        elif j < 2/3:
            cir += gates[1](params[i]).on(qubits)
            ret_param.append(params[i])
        else:
            cir += gates[2](qubits)
            iss += 1
    return cir, ret_param, iss

def fidel(guess, target, resolver):
    pred = cirq.Simulator().simulate(guess, resolver).final_state_vector
    inner = np.inner(np.conj(pred), target) # <V|U>
    fidelity = np.conj(inner) * inner # |<V|U>|^2
    return fidelity.real

def h_opt(param):
    resolve = dict()
    for i in range(len(params)):
        resolve[params[i]] = param[i]
    return 1 - fidel(fake, target_H, resolve) - 0.05 * iss

def x_opt(param):
    resolve = dict()
    for i in range(len(params)):
        resolve[params[i]] = param[i]
    return 1 - fidel(fake, target_X, resolve) - 0.05 * iss

target_H = cirq.Simulator().simulate(create_target_H()).final_state_vector
target_X = cirq.Simulator().simulate(create_target_X()).final_state_vector

iterations = 10
h_costs = []
x_costs = []
min_h = np.inf
min_x = np.inf
cir_x = None
cir_h = None
for i in range(iterations):
    fake, params, iss = initial_ansatz(5)
    if len(params) > 0:
        opt = gp_minimize(h_opt, [(0, 2 * np.pi) for _ in range(len(params))], n_calls=25)
        if opt['fun'] < min_h:
            min_h = opt['fun']
            cir_h = [fake, opt['x']]
    else:
        val = h_opt(None)
        if val < min_h:
            min_h = val
            cir_h = [fake, None]
    fake, params, iss = initial_ansatz(5)
    if len(params) > 0:
        opt = gp_minimize(x_opt, [(0, 2 * np.pi) for _ in range(len(params))], n_calls=25)
        if opt['fun'] < min_x:
            min_x = opt['fun']
            cir_x = [fake, opt['x']]
    else:
        val = x_opt(None)
        if val < min_x:
            min_x = val
            cir_x = [fake, None]
    h_costs.append(min_h)
    x_costs.append(min_x)
    print(i)
    print("H", min_h, cir_h[0], cir_h[1])
    print("X", min_x, cir_x[0], cir_x[1])


plt.plot(h_costs, label='H')
plt.plot(x_costs, label='X')
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.legend()
plt.show()
plt.savefig("one_qubit_w_iss")
