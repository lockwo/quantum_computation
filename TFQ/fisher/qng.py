import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np
import matplotlib.pyplot as plt
from math import prod

def min_loss(y_true, y_pred):
    return y_pred

class QNG(object):
    def __init__(self, num_q, lay, lr):
        super(QNG, self).__init__()
        self.num_qubits = num_q
        self.layers = lay
        self.lr = lr
        self.qubits = [cirq.GridQubit(0, i) for i in range(self.num_qubits)]
        self.num_params = self.num_qubits * self.layers * 3
        self.params = sympy.symbols("params0:%d"%self.num_params)
        self.empty_input = tfq.convert_to_tensor([cirq.Circuit()])
        # run sum
        self.readout_operators = prod([cirq.X(i) for i in self.qubits])
        self.theta = tf.Variable(initial_value=np.random.uniform(0, 2 * np.pi, (self.num_params)), dtype="float32", trainable=True)
        self.qng_model = tfq.layers.ControlledPQC(self.make_circut(), self.readout_operators, differentiator=tfq.differentiators.Adjoint())
        self.gd_model = self.make_model()

    def make_model(self):
        inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
        layer1 = tfq.layers.PQC(self.make_circut(), self.readout_operators, differentiator=tfq.differentiators.Adjoint())(inputs)
        vqc = tf.keras.models.Model(inputs=inputs, outputs=layer1)
        vqc.compile(loss=min_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr))
        vqc.set_weights([tf.identity(self.theta)])
        return vqc

    def make_circut(self):
        cir = cirq.Circuit()
        for i in range(self.layers):
            cir += self.layer(self.params[i * self.num_qubits * 3: (i + 1) * self.num_qubits * 3])
        return cir

    def layer(self, ps):
        c = cirq.Circuit()
        for i in range(self.num_qubits):
            c += cirq.rz(ps[i]).on(self.qubits[i])
        for i in range(self.num_qubits):
            c += cirq.ry(ps[i + self.num_qubits]).on(self.qubits[i])
        for i in range(self.num_qubits):
            c += cirq.rz(ps[i + 2 * self.num_qubits]).on(self.qubits[i])
        for i in range(self.num_qubits - 1):
            c += cirq.CNOT(self.qubits[i], self.qubits[i+1])
        c += cirq.CNOT(self.qubits[-1], self.qubits[0])
        return c

    def gd(self, steps):
        history = self.gd_model.fit(self.empty_input, np.zeros((1, 1)), epochs=steps, batch_size=1)
        plt.plot(history.history['loss'], label='GD')

    def get_circuit(self):
        resolver = cirq.ParamResolver({self.params[i] : self.theta[i].numpy() for i in range(len(self.params))})
        return cirq.resolve_parameters(self.make_circut(), resolver)

    @tf.function
    def fid(self, theta, j1_arr, j2_arr, current_circuit, params, old_theta_circuit):
        values = theta + j1_arr + j2_arr
        values = tf.reshape(values, [1, len(values)])
        fidelity1 = tfq.math.fidelity(current_circuit, tf.convert_to_tensor([s.name for s in params]), values, old_theta_circuit)
        values = theta + j1_arr - j2_arr
        values = tf.reshape(values, [1, len(values)])
        fidelity2 = tfq.math.fidelity(current_circuit, tf.convert_to_tensor([s.name for s in params]), values, old_theta_circuit)
        values = theta - j1_arr + j2_arr
        values = tf.reshape(values, [1, len(values)])
        fidelity3 = tfq.math.fidelity(current_circuit, tf.convert_to_tensor([s.name for s in params]), values, old_theta_circuit)
        values = theta - j1_arr - j2_arr
        values = tf.reshape(values, [1, len(values)])
        fidelity4 = tfq.math.fidelity(current_circuit, tf.convert_to_tensor([s.name for s in params]), values, old_theta_circuit)
        return fidelity1 - fidelity2 - fidelity3 + fidelity4

    def qng(self, steps):
        loss = [self.qng_model([self.empty_input, tf.expand_dims(self.theta, axis=0)])[0][0].numpy()]
        current_circuit = tfq.convert_to_tensor([self.make_circut()])
        for step in range(steps):
            F = np.zeros(shape=(self.num_params, self.num_params))
            print(step, loss[-1])
            old_theta_circuit = tfq.convert_to_tensor([[self.get_circuit()]])
            for j1 in range(self.num_params):
                j1_arr = np.zeros(self.num_params, dtype=np.float32)
                j1_arr[j1] = np.pi/2
                for j2 in range(self.num_params):
                    j2_arr = np.zeros(self.num_params, dtype=np.float32)
                    j2_arr[j2] = np.pi/2
                    F[j1][j2] = self.fid(self.theta, j1_arr, j2_arr, current_circuit, self.params, old_theta_circuit)
            F = -1/8 * F
            F = tf.cast(F, tf.float32)
            with tf.GradientTape() as tape:
                tape.watch(self.theta)
                error = self.qng_model([self.empty_input, tf.expand_dims(self.theta, axis=0)])
            grads = tape.gradient(error, self.theta)
            self.theta = self.theta - tf.squeeze(self.lr * tf.matmul(tf.linalg.pinv(F), tf.reshape(grads, [self.num_params, 1])))
            loss.append(self.qng_model([self.empty_input, tf.expand_dims(self.theta, axis=0)])[0][0].numpy())
        plt.plot(loss, label='QNG')

    def show(self):
        plt.legend()
        plt.show()

if __name__ == "__main__":
    test = QNG(6, 2, 0.001)
    N = 100
    test.gd(N)
    test.qng(N)
    test.show()
