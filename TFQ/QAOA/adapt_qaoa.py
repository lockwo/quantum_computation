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

nodes = 6
complete_graph = nx.complete_graph(nodes)
cycle_graph = nx.cycle_graph(nodes)
cycle_graph.add_edge(0, 3)
cycle_graph.add_edge(1, 4)
cycle_graph.add_edge(2, 5)

#nx.draw(complete_graph)
#plt.show()
#nx.draw(cycle_graph)
#plt.show()

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

def cc(qubits, g):
    c = 0
    for edge in g.edges():
        c += cirq.PauliString(1/2 * cirq.Z(qubits[edge[0]]) * cirq.Z(qubits[edge[1]]) * g.edges[edge]['weight'])
    return c

def make_circuit(p, graph, qs):
    qaoa_circuit = cirq.Circuit()
    num_param = 2 * p 
    qaoa_parameters = sympy.symbols("q0:%d"%num_param)
    for i in qs:
        qaoa_circuit += cirq.H(i)
    for i in range(p):
        qaoa_circuit = cost_hamiltonian(qaoa_circuit, qs, graph, qaoa_parameters[2 * i])
        qaoa_circuit = mixing_hamiltonian(qaoa_circuit, qs, qaoa_parameters[2 * i + 1])
    return qaoa_circuit

def make_qaoa(graph, p):
    qs = [cirq.GridQubit(0, i) for i in range(len(graph.nodes()))]
    qaoa_circuit = make_circuit(p, graph, qs)
    
    cost = cc(qs, graph)
    ins = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)
    outs = tfq.layers.PQC(qaoa_circuit, cost, differentiator=tfq.differentiators.Adjoint())(ins)
    qaoa = tf.keras.models.Model(inputs=ins, outputs=outs)
    return qaoa

def train_qaoa(qaoa):
    opt = tf.keras.optimizers.Adam(learning_rate=0.1)
    tol = 1e-5
    old = np.inf

    initial = cirq.Circuit()
    inputs = tfq.convert_to_tensor([initial])

    while True:
        with tf.GradientTape() as tape:
            error = qaoa(inputs)
        
        grads = tape.gradient(error, qaoa.trainable_variables)
        opt.apply_gradients(zip(grads, qaoa.trainable_variables))
        error = error.numpy()[0][0]
        if abs(old - error) < tol:
            break
        old = error
    
    return error

def generate_circuits(base_cir, ops, sym):
    circuit = []
    op = [tfq.util.exponential([i], [sym]) for i in ops]
    for c in op:
        circuit.append(base_cir + c)
    return circuit

def adapt_qaoa(graph, op_pool, p):
    expectation_layer = tfq.layers.Expectation()
    adapt_iter = 0
    tol = 1e-5

    params = []
    symbols = []
    base_circuit = cirq.Circuit()
    qubits = [cirq.GridQubit(0, i) for i in range(len(graph.nodes()))]
    for q in qubits:
        base_circuit += cirq.H(q)
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.1)
    track = []
    h = cc(qubits, graph)
    counter = 0
    while adapt_iter < p:
        params.append(0.01)
        symbols.append(sympy.symbols("c%d"%adapt_iter))
        base_circuit = cost_hamiltonian(base_circuit, qubits, graph, symbols[-1])

        init = None
        inner_counter = 0
        while True:
            params.append(0.01)
            symbols.append(sympy.symbols("m%d"%counter))
            counter += 1
            inner_counter += 1
            circuits = generate_circuits(base_circuit, op_pool, symbols[-1])
            var = tf.Variable(params, dtype=tf.float32, trainable=True)
            grads = []
            for c in circuits:
                with tf.GradientTape() as tape:
                    tape.watch(var)
                    exp_val = expectation_layer(c, symbol_names=[s.name for s in symbols], symbol_values=[var], operators=h)
                #grads.append(tf.math.abs(tape.gradient(exp_val, var)[-counter:])[0])
                grads.append(tf.math.reduce_mean(tf.math.abs((tape.gradient(exp_val, var)[-inner_counter:]))))
                #grads.append(tf.norm(tape.gradient(exp_val, var)[-counter:]))

            if init is None:
                init = max(grads)
            print(adapt_iter, counter, max(grads), init)
            grads_ = [i for i in grads if i >= init]
            if len(grads_) == 0 or counter > (adapt_iter * 2 + 25):
                break
            else:
                base_circuit = circuits[np.argmax(grads)]

            old = np.inf
            while True:
                with tf.GradientTape() as tape:
                    tape.watch(var)
                    guess = expectation_layer(base_circuit, symbol_names=[s.name for s in symbols], symbol_values=[var], operators=h)
                grads = tape.gradient(guess, var)
                opt.apply_gradients(zip([grads], [var]))
                guess = guess.numpy()[0][0]
                if abs(guess - old) < tol:
                    break
                old = guess
            params = var.numpy().tolist()

        track.append(guess)
        adapt_iter += 1
        print(adapt_iter, track[-1])
    return track

def real_cost(graph):
    sub_lists = []
    for i in range(nodes + 1):
        temp = [list(x) for x in combinations(graph.nodes(), i)]
        sub_lists.extend(temp)

    cut_classic = []
    for sub_list in sub_lists:
        cut_classic.append(nx.algorithms.cuts.cut_size(graph, sub_list))
    
    sol = sub_lists[np.argmax(cut_classic)]

    c = 0
    for edge in graph.edges():
        s1 = -1 if edge[0] in sol else 1
        s2 = -1 if edge[0] in sol else 1
        c += (-1/2 * s1 * s2 * graph.edges[edge]['weight'])
    return c

def s_pool(maxcut_graph):
    pool = []
    cirq_qubits = [cirq.GridQubit(0, i) for i in range(len(maxcut_graph.nodes()))]
    mixing_ham = 0
    for node in maxcut_graph.nodes():
        qubit = cirq_qubits[node]
        mixing_ham += cirq.PauliString(cirq.X(qubit))
    pool.append(mixing_ham)

    mixing_ham = 0
    for node in maxcut_graph.nodes():
        qubit = cirq_qubits[node]
        mixing_ham += cirq.PauliString(cirq.Y(qubit))

    pool.append(mixing_ham)

    for node in maxcut_graph.nodes():
        mixing_ham = 0
        qubit = cirq_qubits[node]
        mixing_ham += cirq.PauliString(cirq.X(qubit))
        pool.append(mixing_ham)

    for node in maxcut_graph.nodes():
        mixing_ham = 0
        qubit = cirq_qubits[node]
        mixing_ham += cirq.PauliString(cirq.Y(qubit))
        pool.append(mixing_ham)
    return pool

def m_pool(maxcut_graph):
    pool = s_pool(maxcut_graph)
    cirq_qubits = [cirq.GridQubit(0, i) for i in range(len(maxcut_graph.nodes()))]
    for pauli in [cirq.X, cirq.Y, cirq.Z]:
        for node1 in range(len(maxcut_graph.nodes())):
            for node2 in range(node1, len(maxcut_graph.nodes())):
                mixing_ham = 0
                qubit1 = cirq_qubits[node1]
                qubit2 = cirq_qubits[node2]
                mixing_ham += cirq.PauliString(pauli(qubit1) * pauli(qubit2))
                pool.append(mixing_ham)

    return pool

graph = cycle_graph
for (u, v) in graph.edges():
    graph.edges[u,v]['weight'] = np.random.uniform(0.1, 2.0)
solution = real_cost(graph)
print(solution)
max_p = 8

single_qubit_pool = s_pool(graph)
multi_qubit_pool = m_pool(graph)

qaoa_energies = []
ps = []

adapt_single = adapt_qaoa(graph, single_qubit_pool, max_p)
adapt_single = [i - solution for i in adapt_single]
adapt_multi = adapt_qaoa(graph, multi_qubit_pool, max_p)
adapt_multi = [i - solution for i in adapt_multi]

for p in range(1, max_p + 1):
    e = train_qaoa(make_qaoa(graph, p))
    print(p, e)
    qaoa_energies.append(e - solution)
    ps.append(p)

plt.plot(ps, qaoa_energies, label="QAOA")
plt.plot(ps, adapt_single, label="ADAPT-single")
plt.plot(ps, adapt_multi, label="ADAPT-multi")
plt.legend()
plt.yscale('log')
plt.ylabel("Energy")
plt.xlabel("p")
plt.show()
