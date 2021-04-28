import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations

def to_dec(x):
    return int("".join(str(i) for i in x), 2) 

nodes = 14
regularity = 6
maxcut_graph = nx.random_regular_graph(n=nodes, d=regularity)

def mixing_hamiltonian(c, qubits, par):
    for i in range(len(qubits)):
        c += cirq.rx(2 * par).on(qubits[i])
    return c

def cost_hamiltonian(c, qubits, g, ps):
    for edge in g.edges():
        c += cirq.CNOT(qubits[edge[0]], qubits[edge[1]])
        c += cirq.rz(ps).on(qubits[edge[1]])
        c += cirq.CNOT(qubits[edge[0]], qubits[edge[1]])
    return c

qs = [cirq.GridQubit(0, i) for i in range(nodes)]
qaoa_circuit = cirq.Circuit()
p = 8

num_param = 2 * p 
qaoa_parameters = sympy.symbols("q0:%d"%num_param)
for i in range(p):
    qaoa_circuit = cost_hamiltonian(qaoa_circuit, qs, maxcut_graph, qaoa_parameters[2 * i])
    qaoa_circuit = mixing_hamiltonian(qaoa_circuit, qs, qaoa_parameters[2 * i + 1])

initial = cirq.Circuit()
for i in qs:
    initial.append(cirq.H(i))

inputs = tfq.convert_to_tensor([initial])

def cc(qubits, g):
    c = 0
    for edge in g.edges():
        c += cirq.PauliString(1/2 * cirq.Z(qubits[edge[0]]) * cirq.Z(qubits[edge[1]]))
    return c

cost = cc(qs, maxcut_graph)
ins = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)
outs = tfq.layers.PQC(qaoa_circuit, cost, differentiator=tfq.differentiators.Adjoint())(ins)
qaoa = tf.keras.models.Model(inputs=ins, outputs=outs)
opt = tf.keras.optimizers.Adam(lr=0.01)

losses = []
epoch = 0
tol = 1e-4
old = 100
while True:
    with tf.GradientTape() as tape:
        error = qaoa(inputs)
    
    grads = tape.gradient(error, qaoa.trainable_variables)
    opt.apply_gradients(zip(grads, qaoa.trainable_variables))
    error = error.numpy()[0][0]
    losses.append(error)
    if epoch % 10 == 0:
        print(epoch, error)
    if abs(old - error) < tol:
        break
    old = error
    epoch += 1

plt.plot(losses)
plt.title("QAOA with TFQ")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()
plt.savefig("qaoa")

params = qaoa.trainable_variables
print(params)

sample_circuit = tfq.layers.AddCircuit()(inputs, append=qaoa_circuit)
output = tfq.layers.Sample()(sample_circuit, symbol_names=qaoa_parameters, symbol_values=params, repetitions=1000)

quantum_preds = []
data = []
for bits in output.values:
    temp = []
    data.append(to_dec(bits.numpy()))
    for pos, bit in enumerate(bits):
        if bit == 1:
            temp.append(pos)
    quantum_preds.append(temp)

sub_lists = []
for i in range(nodes + 1):
    temp = [list(x) for x in combinations(maxcut_graph.nodes(), i)]
    sub_lists.extend(temp)

cut_classic = []
for sub_list in sub_lists:
  cut_classic.append(nx.algorithms.cuts.cut_size(maxcut_graph,sub_list))

cut_quantum = []
for cut in quantum_preds:
  cut_quantum.append(nx.algorithms.cuts.cut_size(maxcut_graph,cut))

print(np.mean(cut_quantum), np.max(cut_classic))
print(np.mean(cut_quantum)/np.max(cut_classic))
