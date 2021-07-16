import numpy as np
import random
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
from replay_buffer import ReplayBuffer

class Hybrid(tf.keras.layers.Layer):
    def __init__(self, q_params, circuit, ops, encoder, out, actions) -> None:
        super().__init__()
        self.pre_net = encoder
        diff = tfq.differentiators.Adjoint()
        self.quantum_operation = tfq.layers.ControlledPQC(circuit, ops, differentiator=diff)
        self.outs = out
        if out == "d":
            self.post_net = tf.keras.layers.Dense(actions)
        num_q_params = len(q_params)
        self.quantum_weights = tf.Variable(initial_value=np.random.uniform(0, 2 * np.pi, num_q_params), dtype="float32", trainable=True)
        self.circuit_tensor = tfq.convert_to_tensor([cirq.Circuit()])
    
    def call(self, inputs):
        circuit_batch_dim = tf.gather(tf.shape(inputs), 0)
        net_outputs = self.pre_net(inputs)
        tiled_b = tf.tile(tf.expand_dims(self.quantum_weights, 0), [circuit_batch_dim, 1])
        quantum_inputs = tf.concat([net_outputs, tiled_b], axis=1)
        tiled_circuits = tf.tile(self.circuit_tensor, [circuit_batch_dim])
        quantum_output = self.quantum_operation([tiled_circuits, quantum_inputs])
        if self.outs == "d":
            return self.post_net(quantum_output)
        return quantum_output

class DQN_Q(object):
    def __init__(self, action_size, enc, qu, ou):
        # Quantum
        self.encoder = enc
        self.q = qu
        self.qubits = [cirq.GridQubit(0, i) for i in range(self.q)]
        self.output = ou
        # Classical 
        self.action_space = action_size
        self.q_network = self.make_net()
        self.q_target = self.make_net()
        self.move_weights()
        self.buff = 1000000
        self.rb = ReplayBuffer(self.buff)
        self.batch = 32
        self.gamma = 0.99 
        self.epsilon = 1.0
        self.epsilon_decay_frames = 0.5 * 1e6
        self.epsilon_min = 0.02
        self.learning_rate = 3e-4
        self.opt = tf.keras.optimizers.Adam(lr=self.learning_rate)
        self.tau = 0.001
        self.iter = 0
        self.training_frequency = 4
        self.counter = 0
        self.update = 10000

    def move_weights(self):
        self.q_target.set_weights(self.q_network.get_weights())

    # Next few functions from https://www.tensorflow.org/quantum/tutorials/qcnn
    def one_qubit_unitary(self, bit, symbols):
        return cirq.Circuit(
            cirq.X(bit)**symbols[0],
            cirq.Y(bit)**symbols[1],
            cirq.Z(bit)**symbols[2])

    def two_qubit_unitary(self, bits, symbols):
        circuit = cirq.Circuit()
        circuit += self.one_qubit_unitary(bits[0], symbols[0:3])
        circuit += self.one_qubit_unitary(bits[1], symbols[3:6])
        circuit += [cirq.ZZ(*bits)**symbols[6]]
        circuit += [cirq.YY(*bits)**symbols[7]]
        circuit += [cirq.XX(*bits)**symbols[8]]
        circuit += self.one_qubit_unitary(bits[0], symbols[9:12])
        circuit += self.one_qubit_unitary(bits[1], symbols[12:])
        return circuit

    def two_qubit_pool(self, source_qubit, sink_qubit, symbols):
        pool_circuit = cirq.Circuit()
        sink_basis_selector = self.one_qubit_unitary(sink_qubit, symbols[0:3])
        source_basis_selector = self.one_qubit_unitary(source_qubit, symbols[3:6])
        pool_circuit.append(sink_basis_selector)
        pool_circuit.append(source_basis_selector)
        pool_circuit.append(cirq.CNOT(control=source_qubit, target=sink_qubit))
        pool_circuit.append(sink_basis_selector**-1)
        return pool_circuit

    def quantum_pool_circuit(self, source_bits, sink_bits, symbols):
        circuit = cirq.Circuit()
        for source, sink in zip(source_bits, sink_bits):
            circuit += self.two_qubit_pool(source, sink, symbols)
        return circuit

    def quantum_conv_circuit(self, bits, symbols):
        circuit = cirq.Circuit()
        for first, second in zip(bits[0::2], bits[1::2]):
            circuit += self.two_qubit_unitary([first, second], symbols)
        for first, second in zip(bits[1::2], bits[2::2] + [bits[0]]):
            circuit += self.two_qubit_unitary([first, second], symbols)
        return circuit

    def qdense(self, params, qubits):
        # 2 * (3 * q + 3 * (q - 2))
        # 48, 108, 168
        l = cirq.Circuit()
        q = len(qubits)
        l.append([cirq.Moment([cirq.rx(params[j]).on(qubits[j]) for j in range(q)])]) # 1, 2, 3, 4, 5
        l.append([cirq.Moment([cirq.ry(params[j + q]).on(qubits[j]) for j in range(q)])]) # 6 - 10
        l.append([cirq.Moment([cirq.rz(params[j + 2 * q]).on(qubits[j]) for j in range(q)])]) # 10 - 15
        l.append(cirq.CNOT(qubits[0], qubits[1]))
        l.append(cirq.CNOT(qubits[-2], qubits[-1]))
        l.append([cirq.Moment([cirq.rx(params[j - 1 + 3 * q]).on(qubits[j]) for j in range(1, q - 1)])]) 
        l.append([cirq.Moment([cirq.ry(params[j - 1 + 3 * q + q - 2]).on(qubits[j]) for j in range(1, q - 1)])])
        l.append([cirq.Moment([cirq.rz(params[j - 1 + 3 * q + 2 * (q - 2)]).on(qubits[j]) for j in range(1, q - 1)])])
        for i in range(1, q - 2):
            l.append(cirq.CNOT(qubits[i], qubits[i+1]))
        return l

    def make_encoder(self, qubits, symbols):
        m = cirq.Circuit()
        q = len(qubits)
        u = 3 * q + 3 * (q - 2)
        m += self.qdense(symbols[:u], qubits)
        m += self.qdense(symbols[u:2 * u], qubits)
        return m

    def make_net(self):
        if self.q == 5:
            # 48 (encode) + 15 (conv) + 24 (ending) = 5
            symbols = sympy.symbols('q0:87')
            controlled_params = symbols[:48]
            free_params = symbols[48:]
            quantum_encoder = self.make_encoder(self.qubits, controlled_params)
            readout_ops = [cirq.Z(self.qubits[i]) for i in range(min(self.action_space, 5))]
            conv = self.quantum_conv_circuit(self.qubits, free_params[:15])
            ending = self.qdense(free_params[15:], self.qubits)
        elif self.q == 10:
            # 108 (encode) + 30 (conv) + 6 (pool) + 24 (ending) = 5
            symbols = sympy.symbols('q0:168')
            controlled_params = symbols[:108]
            free_params = symbols[108:]
            quantum_encoder = self.make_encoder(self.qubits, controlled_params)
            readout_ops = [cirq.Z(self.qubits[i]) for i in range(min(self.action_space, 5))]
            conv = self.quantum_conv_circuit(self.qubits, free_params[:15])
            conv += self.quantum_pool_circuit(self.qubits[5:], self.qubits[:5], free_params[15:21])
            conv += self.quantum_conv_circuit(self.qubits[:5], free_params[21:36])
            ending = self.qdense(free_params[36:], self.qubits[:5])
        elif self.q  == 15:
            # 168 (encode) + 30 (conv) + 6 (pool) +  30 (ending) = 6
            symbols = sympy.symbols('q0:234')
            controlled_params = symbols[:168]
            free_params = symbols[168:]
            quantum_encoder = self.make_encoder(self.qubits, controlled_params)
            readout_ops = [cirq.Z(self.qubits[i]) for i in range(min(self.action_space, 6))]
            conv = self.quantum_conv_circuit(self.qubits, free_params[:15])
            conv += self.quantum_pool_circuit(self.qubits[7:14], self.qubits[:7], free_params[15:21])
            conv += self.quantum_conv_circuit(self.qubits[:7], free_params[21:36])
            ending = self.qdense(free_params[36:], self.qubits[:6])
        
        classical_encoder = self.controller(self.encoder)
        #classical_encoder.summary()
        full_circuit = quantum_encoder + conv + ending
        #print("Encoder", quantum_encoder)
        #print("Conv", conv)
        #print("Ending", ending)
        commands_input = tf.keras.layers.Input(shape=(84, 84, 4))
        main_layer = Hybrid(free_params, full_circuit, readout_ops, classical_encoder, self.output, self.action_space)(commands_input)
        model = tf.keras.Model(inputs=commands_input, outputs=main_layer)
        #model.summary()
        return model

    def controller(self, type):
        u = 2 * (3 * len(self.qubits) + 3 * (len(self.qubits) - 2))
        if type == "d":
            inputs = tf.keras.layers.Input(shape=(84, 84, 4))
            x = tf.keras.layers.Flatten()(inputs)
            x = tf.keras.layers.Dense(3, activation='relu')(x)
            x = tf.keras.layers.Dense(50, activation='relu')(x)
            x = tf.keras.layers.Dense(u)(x)
            model = tf.keras.models.Model(inputs=inputs, outputs=x)
        elif type == "c":
            inputs = tf.keras.layers.Input(shape=(84, 84, 4))
            x = tf.keras.layers.Conv2D(32, 8, strides=(6,6), activation='relu')(inputs)
            x = tf.keras.layers.Conv2D(64, 4, strides=(4,4), activation='relu')(x)
            x = tf.keras.layers.Conv2D(64, 3, strides=(2,2), activation='relu')(x)
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(u)(x)
            model = tf.keras.models.Model(inputs=inputs, outputs=x)
        return model

    def remember(self, state, action, reward, next_state, done):
        self.rb.add(state, action, reward, next_state, done)
        self.counter += 1

    def get_action(self, obs):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= 1/self.epsilon_decay_frames
        if random.random() < self.epsilon: 
            return np.random.choice(min(self.action_space, 5))
        else:
            test = np.expand_dims(np.array(obs)/255.0, axis=0)
            act = self.q_network(test).numpy()[0]
            act = np.argmax(act)
            return act 

    @tf.function
    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    @tf.function
    def grad_update(self, state_batch, action_batch, reward_batch, next_state_batch, dones_batch):
        with tf.GradientTape() as tape:
            next_q = self.q_target(next_state_batch)
            next_q = tf.expand_dims(tf.reduce_max(next_q, axis=1), -1)
            y = reward_batch + (1 - dones_batch) * self.gamma * next_q
            q = self.q_network(state_batch, training=True)
            pred = tf.gather_nd(q, action_batch)
            pred = tf.reshape(pred, [self.batch, 1])
            msbe = tf.math.reduce_mean(tf.math.square(y - pred))
        grads = tape.gradient(msbe, self.q_network.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.q_network.trainable_variables))

    def train(self):
        states, acts, res, nexts, dones = self.rb.sample(self.batch)
        states = states / 255.0
        nexts = nexts / 255.0
        state_batch = tf.convert_to_tensor(states)
        action_batch = tf.convert_to_tensor(acts, dtype=tf.int32)
        action_batch = [[i, action_batch[i].numpy()] for i in range(len(action_batch))]
        reward_batch = tf.convert_to_tensor(res, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(nexts)
        dones_batch = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        reward_batch = tf.reshape(reward_batch, [len(reward_batch), 1])
        dones_batch = tf.reshape(dones_batch, [len(dones_batch), 1])

        self.grad_update(state_batch, action_batch, reward_batch, next_state_batch, dones_batch)

        if self.iter % self.update == 0:
            self.update_target(self.q_target.trainable_variables, self.q_network.trainable_variables, 1)
        else:
            self.update_target(self.q_target.trainable_variables, self.q_network.trainable_variables, self.tau)

        self.iter += 1
