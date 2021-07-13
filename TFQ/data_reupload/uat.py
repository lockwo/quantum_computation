import tensorflow as tf
import tensorflow_quantum as tfq 
import numpy as np
import cirq
import sympy


class UAT(tf.keras.layers.Layer):
    def __init__(self, depth, unitary_type) -> None:
        super(UAT, self).__init__()
        self.num_each_params = depth
        self.layers = depth
        self.type = unitary_type
        if self.type == "fourier":
            self.alpha = tf.Variable(initial_value=np.random.uniform(0, 2 * np.pi, (1, self.num_each_params)), dtype="float32", trainable=True)
            self.beta = tf.Variable(initial_value=np.random.uniform(0, 2 * np.pi, (1, self.num_each_params)), dtype="float32", trainable=True)
            self.psi = tf.Variable(initial_value=np.random.uniform(0, 2 * np.pi, (1, self.num_each_params)), dtype="float32", trainable=True)
            self.lambd = tf.Variable(initial_value=np.random.uniform(0, 2 * np.pi, (1, self.num_each_params)), dtype="float32", trainable=True)
            self.w = tf.Variable(initial_value=np.random.uniform(0, 2 * np.pi, (1, self.num_each_params)), dtype="float32", trainable=True)
            self.params_gates = 5 * self.num_each_params
            self.g = 5
        else:
            self.alpha = tf.Variable(initial_value=np.random.uniform(0, 2 * np.pi, (1, self.num_each_params)), dtype="float32", trainable=True)
            self.psi = tf.Variable(initial_value=np.random.uniform(0, 2 * np.pi, (1, self.num_each_params)), dtype="float32", trainable=True)
            self.w = tf.Variable(initial_value=np.random.uniform(0, 2 * np.pi, (1, self.num_each_params)), dtype="float32", trainable=True)
            self.params_gates = 2 * self.num_each_params
            self.g = 2
        self.qubit = cirq.GridQubit(0, 0)
        self.params = sympy.symbols("params0:%d"%self.params_gates)
        self.cpqc = tfq.layers.ControlledPQC(self.make_circuit(), [cirq.Z(self.qubit)], differentiator=tfq.differentiators.Adjoint())
        self.in_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.indices = []
        i = 0
        while i < self.num_each_params:
            for j in range(self.g):
                self.indices.append(i + self.num_each_params * j)
            i += 1

    def make_circuit(self):
        cir = cirq.Circuit()
        if self.type == "fourier":
            for i in range(self.layers):
                cir += cirq.rz(self.params[i * 5]).on(self.qubit)
                cir += cirq.ry(self.params[i * 5 + 1]).on(self.qubit)
                cir += cirq.rz(self.params[i * 5 + 2]).on(self.qubit)
                cir += cirq.rz(self.params[i * 5 + 3]).on(self.qubit)
                cir += cirq.ry(self.params[i * 5 + 4]).on(self.qubit)
        else:
            for i in range(self.layers):
                cir += cirq.ry(self.params[i * 2]).on(self.qubit)
                cir += cirq.rz(self.params[i * 2 + 1]).on(self.qubit)
        return cir

    # inputs = (batch, in_size)
    def call(self, inputs):
        if self.type == "fourier":
            return self.fourier_call(inputs)
        else:
            return self.uat_call(inputs)

    def fourier_call(self, ins):
        num_batch = tf.gather(tf.shape(ins), 0)
        # (1, 1) -> (batch, 1)
        input_circuits = tf.repeat(self.in_circuit, repeats=num_batch)
        # (batch, in_size) -> (batch, num_each_params)
        inputs = tf.tile(ins, [1, int(self.num_each_params/ins.shape[1])])
        # (1, num_each_params) -> (batch, num_each_params)
        a = tf.tile(self.alpha, [num_batch, 1])
        b = tf.tile(self.beta, [num_batch, 1])
        w = tf.tile(self.w, [num_batch, 1])
        psi = tf.tile(self.psi, [num_batch, 1])
        lamb = tf.tile(self.lambd, [num_batch, 1])
        # (batch, num_each_params) * (batch, num_each_params) -> (batch, num_each_params)
        aplusb = a + b
        lamb = 2 * lamb
        aminusb = a - b
        wx = 2 * w * inputs
        psi = 2 * psi
        # (batch, num_each_params) -> (batch, params_gates)
        full_params = tf.concat([aplusb, lamb, aminusb, wx, psi], axis=1)
        full_params = tf.gather(full_params, self.indices, axis=1)
        return self.cpqc([input_circuits, full_params])

    # Different from paper, UAT only accepts 1D (not 1-3D) inputs
    def uat_call(self, ins):
        num_batch = tf.gather(tf.shape(ins), 0)
        # (1, 1) -> (batch, 1)
        input_circuits = tf.repeat(self.in_circuit, repeats=num_batch)
        # (batch, in_size) -> (batch, num_each_params)
        inputs = tf.tile(ins, [1, int(self.num_each_params/ins.shape[1])])
        # (1, num_each_params) -> (batch, num_each_params)
        psi = tf.tile(self.psi, [num_batch, 1])
        w = tf.tile(self.w, [num_batch, 1])
        a = tf.tile(self.alpha, [num_batch, 1])
        # (batch, num_each_params) * (batch, num_each_params) -> (batch, num_each_params)
        psi = 2 * psi
        wxa = 2 * w * inputs + 2 * a
        # (batch, num_each_params) -> (batch, params_gates)
        full_params = tf.concat([psi, wxa], axis=1)
        full_params = tf.gather(full_params, self.indices, axis=1)
        return self.cpqc([input_circuits, full_params])
