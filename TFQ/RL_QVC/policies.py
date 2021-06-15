import tensorflow_quantum as tfq
import cirq 
import sympy
import numpy as np
import tensorflow as tf

class ReUpPolicy(tf.keras.layers.Layer):
    def __init__(self, num_q, lays, num_actions) -> None:
        super(ReUpPolicy, self).__init__()
        self.qubits = [cirq.GridQubit(0, i) for i in range(num_q)]
        self.num_params = 2 * lays * len(self.qubits)
        self.phi = tf.Variable(initial_value=np.random.uniform(0, 2 * np.pi, (1, self.num_params)), dtype="float32", trainable=True)
        self.lamb = tf.Variable(initial_value=np.ones((1, self.num_params)), dtype="float32", trainable=True)
        self.w = tf.Variable(initial_value=np.random.uniform(0, 2 * np.pi, (len(self.qubits), num_actions)), dtype="float32", trainable=True)
        self.total_params = self.num_params * 2
        self.params = sympy.symbols("params0:%d"%self.total_params)
        self.readout_ops = [cirq.Z(i) for i in self.qubits]
        self.model = tfq.layers.ControlledPQC(self.make_circuit(lays, self.params), self.readout_ops, differentiator=tfq.differentiators.Adjoint())
        self.in_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.beta = 1
        # [phis, lambs]
        self.indices = []
        i = 0
        while i < self.num_params:
            for j in range(len(self.qubits) * 2):
                self.indices.append(i + j)
            for j in range(len(self.qubits) * 2):
                self.indices.append(i + self.num_params + j)
            i += len(self.qubits) * 2

    def make_circuit(self, lays, params):
        cir = cirq.Circuit()
        for i in self.qubits:
            cir += cirq.H(i)

        params_per_layer = 2 * 2 * len(self.qubits)
        p = 0
        for i in range(lays):
            cir += self.u_ent(params[p:p + params_per_layer//2])
            cir += self.u_enc(params[p + params_per_layer//2:p + params_per_layer])
            p += params_per_layer
            if i == 0:
                print(cir)

        return cir

    def u_ent(self, ps):
        c = cirq.Circuit()
        for i in range(len(self.qubits)):
            c += cirq.rz(ps[i]).on(self.qubits[i])
        for i in range(len(self.qubits)):
            c += cirq.ry(ps[i + len(self.qubits)]).on(self.qubits[i])
        for i in range(len(self.qubits) - 1):
            c += cirq.CZ(self.qubits[i], self.qubits[i+1])
        c += cirq.CZ(self.qubits[-1], self.qubits[0])
        return c

    def u_enc(self, ps):
        c = cirq.Circuit()
        for i in range(len(self.qubits)):
            c += cirq.ry(ps[i]).on(self.qubits[i])
        for i in range(len(self.qubits)):
            c += cirq.rz(ps[i + len(self.qubits)]).on(self.qubits[i])
        return c

    # inputs = (batch, in_size)
    def call(self, inputs):
        num_batch = tf.gather(tf.shape(inputs), 0)
        # (1, 1) -> (batch, 1)
        input_circuits = tf.repeat(self.in_circuit, repeats=num_batch)
        # (batch, in_size) -> (batch, num_params)
        inputs = tf.tile(inputs, [1, int(self.num_params/inputs.shape[1])])
        # (1, num_param) * (batch, num_params) -> (batch, num_params)
        lambs = tf.math.multiply(self.lamb, inputs)
        # (1, num_param) -> (batch, num_params)
        phis = tf.tile(self.phi, [num_batch, 1])
        # (batch, num_params), (batch, num_params) -> (batch, total_params)
        full_params = tf.concat([phis, lambs], axis=1)
        full_params = tf.gather(full_params, self.indices, axis=1)
        # -> (batch, n_qubit)
        output = self.model([input_circuits, full_params])
        # (batch, n_qubit) -> (batch, n_act)
        logits = tf.linalg.matmul(output*self.beta, self.w)
        return tf.nn.softmax(logits)


class NoReUpPolicy(tf.keras.layers.Layer):
    def __init__(self, num_q, lays, num_actions) -> None:
        super(NoReUpPolicy, self).__init__()
        self.qubits = [cirq.GridQubit(0, i) for i in range(num_q)]
        self.num_params = 2 * lays * len(self.qubits)
        self.phi = tf.Variable(initial_value=np.random.uniform(0, 2 * np.pi, (1, self.num_params)), dtype="float32", trainable=True)
        self.lamb = tf.Variable(initial_value=np.ones((1, 2 * len(self.qubits))), dtype="float32", trainable=True)
        self.w = tf.Variable(initial_value=np.random.uniform(0, 2 * np.pi, (len(self.qubits), num_actions)), dtype="float32", trainable=True)
        self.total_params = self.num_params + 2 * len(self.qubits)
        self.params = sympy.symbols("params0:%d"%self.total_params)
        self.readout_ops = [cirq.Z(i) for i in self.qubits]
        self.model = tfq.layers.ControlledPQC(self.make_circuit(lays, self.params), self.readout_ops, differentiator=tfq.differentiators.Adjoint())
        self.in_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.beta = 1

    def make_circuit(self, lays, params):
        cir = cirq.Circuit()
        for i in self.qubits:
            cir += cirq.H(i)

        params_per_layer = 2 * len(self.qubits)
        cir += self.u_enc(params[0:params_per_layer])
        p = params_per_layer
        for i in range(lays):
            cir += self.u_ent(params[p:p + params_per_layer])
            p += params_per_layer
            if i == 0:
                print(cir)

        return cir

    def u_ent(self, ps):
        c = cirq.Circuit()
        for i in range(len(self.qubits)):
            c += cirq.rz(ps[i]).on(self.qubits[i])
        for i in range(len(self.qubits)):
            c += cirq.ry(ps[i + len(self.qubits)]).on(self.qubits[i])
        for i in range(len(self.qubits) - 1):
            c += cirq.CZ(self.qubits[i], self.qubits[i+1])
        c += cirq.CZ(self.qubits[-1], self.qubits[0])
        return c

    def u_enc(self, ps):
        c = cirq.Circuit()
        for i in range(len(self.qubits)):
            c += cirq.ry(ps[i]).on(self.qubits[i])
        for i in range(len(self.qubits)):
            c += cirq.rz(ps[i + len(self.qubits)]).on(self.qubits[i])
        return c

    # inputs = (batch, in_size)
    def call(self, inputs):
        num_batch = tf.gather(tf.shape(inputs), 0)
        # (1, 1) -> (batch, 1)
        input_circuits = tf.repeat(self.in_circuit, repeats=num_batch)
        # (batch, in_size) -> (batch, 2 * num_qubits)
        inputs = tf.tile(inputs, [1, int((2 * len(self.qubits))/inputs.shape[1])])
        # (1, 2 * num_qubits) * (batch, 2 * num_qubits) -> (batch, 2 * num_qubits)
        lambs = tf.math.multiply(self.lamb, inputs)
        # (1, num_param) -> (batch, num_params)
        phis = tf.tile(self.phi, [num_batch, 1])
        # (batch, 2 * num_qubits), (batch, num_params) -> (batch, total_params)
        full_params = tf.concat([lambs, phis], axis=1)
        # -> (batch, n_qubit)
        output = self.model([input_circuits, full_params])
        # (batch, n_qubit) -> (batch, n_act)
        logits = tf.linalg.matmul(output*self.beta, self.w)
        return tf.nn.softmax(logits)
