import networkx as nx
from matplotlib import pyplot as plt
import cirq
import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import random

'''
The MaxCut problem is a well-known optimization problem in which the nodes of a given undirected graph have to be divided in two sets (often referred as the set of “white” and “black” nodes) 
such that the number of edges connecting a white node with a black node are maximized. The MaxCut problem is a problem on which the QAOA algorithm has proved to be useful 
(for an explanation of the QAOA algorithm you can read this blogpost).

At this link you can find an explicit implementation of the QAOA algorithm to solve the MaxCut problem for the simpler case of an unweighted graph. 
We ask you to generalize the above code to include also the solution for the case of weighted graphs. 
You can use the same code or you can also do an alternative implementation using, for example, qiskit. 
The important point is that you do not make use of any built-in QAOA functionalities.
'''

nodes = 8
G = nx.erdos_renyi_graph(n=nodes, p=0.6)

weighted = True

for (u, v) in G.edges():
    if weighted:
        w = random.uniform(0, 1)
    else:
        w = 1
    G.edges[u,v]['weight'] = w

#nx.draw(G)
#plt.show()
#plt.clf()

# Defines the list of qubits
depth = 8
rep = 1000
qubits = [cirq.GridQubit(0, i) for i in range(nodes)]

# Defines the initialization
def initialization(cir, qubits):
    for i in qubits:
        cir.append(cirq.H(i))

# Defines the cost unitary
def cost_unitary(cir, qubits, gamma):
    for i in G.edges():
        cir.append(cirq.CNOT(qubits[i[0]], qubits[i[1]]))
        cir.append(cirq.rz(gamma).on(qubits[i[1]]))
        cir.append(cirq.CNOT(qubits[i[0]], qubits[i[1]]))
        
# Defines the mixer unitary
def mixer_unitary(cir, qubits, beta):
    for i in qubits:
        cir.append(cirq.rx(2 * beta).on(i))

# Executes the circuit
def create_circuit(params):
    gammas = [j for i, j in enumerate(params) if i % 2 == 0]
    betas = [j for i, j in enumerate(params) if i % 2 == 1]
    #print(gamma, alpha)
    
    circuit = cirq.Circuit()
    initialization(circuit, qubits)
    for i in range(depth):
        cost_unitary(circuit, qubits, gammas[i])
        mixer_unitary(circuit, qubits, betas[i])
    circuit.append(cirq.measure(*qubits, key='x'))

    simulator = cirq.Simulator()
    results = simulator.run(circuit, repetitions=rep)
    results = str(results)[2:].split(", ")
    #print(results)
    new_res = []
    for i in range(rep):
        hold = []
        for j in range(nodes):
            hold.append(int(results[j][i]))
        new_res.append(hold)
        #print(hold)

    return new_res

# Defines the cost function
def cost_function(params):
    av = create_circuit(params)
    total_cost = 0
    for i in range(len(av)):
        for j in G.edges():
            total_cost += G.get_edge_data(j[0], j[1])['weight'] * 0.5 * (((1 - 2 * av[i][j[0]]) * (1 - 2 * av[i][j[1]])) - 1)
    total_cost = float(total_cost)/rep
    return total_cost

# Defines the optimization method
optimal_params = None
optimal_val = np.inf

#create_circuit(np.random.uniform(-np.pi, np.pi, 2 * depth))
#input()
for i in range(8):
    init = np.random.uniform(-np.pi, np.pi, 2 * depth)
    out = minimize(cost_function, x0=init, method="COBYLA", options={'maxiter':200})
    print(i, out['fun'])
    if out['fun'] < optimal_val:
        optimal_params = out['x']
        optimal_val = out['fun']    

f = create_circuit(optimal_params)

quantum_preds = []
for bits in f:
    temp = []
    for pos, bit in enumerate(bits):
        if bit == 1:
            temp.append(pos)
    quantum_preds.append(temp)

sub_lists = []
for i in range(nodes + 1):
    temp = [list(x) for x in combinations(G.nodes(), i)]
    sub_lists.extend(temp)

cut_classic = []
for sub_list in sub_lists:
    cut_classic.append(nx.algorithms.cuts.cut_size(G, sub_list, weight='weight'))

cut_quantum = []
for cut in quantum_preds:
  cut_quantum.append(nx.algorithms.cuts.cut_size(G, cut, weight='weight'))

print(np.mean(cut_quantum), np.max(cut_classic))
print(np.mean(cut_quantum)/np.max(cut_classic))
