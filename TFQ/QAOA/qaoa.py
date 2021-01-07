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

tf.random.set_seed(864)

nodes = 10
regularity = 3

dic = {0: [3, 8, 2],
 1: [2, 4, 7],
 2: [1, 8, 0],
 3: [9, 6, 0],
 4: [7, 6, 1],
 5: [9, 7, 8],
 6: [9, 4, 3],
 7: [4, 5, 1],
 8: [2, 0, 5],
 9: [5, 6, 3]}

maxcut_graph = nx.from_dict_of_lists(dic)

#nodes = 4
#regularity = 2
#maxcut_graph = nx.random_regular_graph(n=nodes, d=regularity)

nx.draw_networkx(maxcut_graph)
plt.show()  

qubits = [cirq.GridQubit(0, i) for i in range(nodes)]

initial = cirq.Circuit()
for i in qubits:
    initial.append(cirq.H(i))

mixing_hamiltonian = 0
for i in range(len(qubits)):
    mixing_hamiltonian += cirq.PauliString(cirq.X(qubits[i]))

cost_hamiltonian = maxcut_graph.number_of_edges()/2
for edge in maxcut_graph.edges():
    cost_hamiltonian += cirq.PauliString(1/2 * cirq.Z(qubits[edge[0]]) * cirq.Z(qubits[edge[1]]))

qaoa_circuit = cirq.Circuit()
p = 1  
qaoa_parameters = []
for i in range(p):
    name = "a" * (i + 1)
    cost_parameter = sympy.symbols(name)
    name = "b" * (i + 1)
    mixing_parameter = sympy.symbols(name)
    qaoa_parameters.append(cost_parameter)
    qaoa_parameters.append(mixing_parameter)
    qaoa_circuit += tfq.util.exponential(operators = [cost_hamiltonian, mixing_hamiltonian], coefficients=[cost_parameter, mixing_parameter])

inputs = tfq.convert_to_tensor([initial])

ins = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)
outs = tfq.layers.PQC(qaoa_circuit, cost_hamiltonian)(ins)
qaoa = tf.keras.models.Model(inputs=ins, outputs=outs)
qaoa.compile(loss=tf.keras.losses.MAE, optimizer=tf.keras.optimizers.Adam())
optimal = np.array([0])
history = qaoa.fit(inputs, optimal, epochs=800)

plt.plot(history.history['loss'])
plt.title("QAOA with TFQ")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

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

xticks = range(0, 2**nodes)
xtick_labels = list(map(lambda x: format(x, "04b"), xticks))
bins = np.arange(0, 2**nodes + 1) - 0.5
plt.xticks(xticks, xtick_labels, rotation="vertical")
plt.hist(data, bins=bins)
plt.show()
