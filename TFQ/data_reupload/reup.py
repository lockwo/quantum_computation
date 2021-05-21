import tensorflow as tf
import tensorflow_quantum as tfq 
import numpy as np
import cirq
import sympy

class ReUploadPQC(tf.keras.layers.Layer):
    def __init__(self, qubit, layers, obs) -> None:
        super(ReUploadPQC, self).__init__()
        self.num_params = len(qubit) * 3 * layers
        self.layers = layers
        self.qubits = qubit
        self.theta = tf.Variable(initial_value=np.random.uniform(0, 2 * np.pi, (1, self.num_params)), dtype="float32", trainable=True)
        self.w = tf.Variable(initial_value=np.random.uniform(0, 2 * np.pi, (1, self.num_params)), dtype="float32", trainable=True)
        self.params = sympy.symbols("params0:%d"%self.num_params)
        self.model = tfq.layers.ControlledPQC(self.make_circuit(layers, self.params), obs, differentiator=tfq.differentiators.Adjoint())
        self.in_circuit = tfq.convert_to_tensor([cirq.Circuit()])

    def make_circuit(self, layers, params):
        c = cirq.Circuit()
        for i in range(layers):
            c = self.layer(c, params[len(self.qubits) * i * 3: (i * 3 + 3) * len(self.qubits)])
        return c

    def layer(self, cir, params):
        for i in range(len(self.qubits)):
            cir += cirq.ry(params[i*3]).on(self.qubits[i])
            cir += cirq.rz(params[i*3 + 1]).on(self.qubits[i])
            cir += cirq.ry(params[i*3 + 2]).on(self.qubits[i])
            if len(self.qubits) > 1:
                cir += cirq.CNOT(self.qubits[i], self.qubits[(i + 1) % len(self.qubits)])
        return cir

    # inputs = (batch, in_size)
    def call(self, inputs):
        num_batch = tf.gather(tf.shape(inputs), 0)
        # (1, 1) -> (batch, 1)
        input_circuits = tf.repeat(self.in_circuit, repeats=num_batch)
        # (batch, in_size) -> (batch, num_params)
        inputs = tf.tile(inputs, [1, int(self.num_params/inputs.shape[1])])
        # (1, num_param) * (batch, num_params) -> (batch, num_params)
        w = tf.math.multiply(self.w, inputs)
        # (1, num_param) -> (batch, num_params)
        thetas = tf.tile(self.theta, [num_batch, 1])
        # (batch, num_params) + (batch, num_params) -> (batch, num_params)
        params = thetas + w
        return self.model([input_circuits, params])
