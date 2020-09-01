import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import tensorflow as tf
import tensorflow_quantum as tfq
from collections import deque
import cirq 
import sympy

#tf.compat.v1.disable_eager_execution()

class A2C_agent(object):
    def __init__(self, action_size, state_size):
        self.action_space = action_size
        self.state_space = state_size[0]
        self.qbits = self.make_bits(4)
        self.actor, self.critic = self.make_net()
        # Q Learning Parameters
        self.gamma = 0.95 # DISCOUNT FACTOR, CLOSE TO 1 = LONG TERM
        self.states, self.rewards, self.values, self.actions = [], [], [], []

    def _entropy_loss(self, target, output):
        return -tf.math.reduce_mean(target*tf.math.log(tf.clip_by_value(output,1e-10,1-1e-10)))

    def make_bits(self, num):
        bits = []
        for i in range(num):
            bits.append(cirq.GridQubit(0, i))
        return bits

    def make_net(self):
        #diff = tfq.differentiators.ParameterShift()
        diff = tfq.differentiators.SGDifferentiator()
        readout_operators0 = [cirq.Z(self.qbits[2]), cirq.Z(self.qbits[3])]
        inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
        actor = tfq.layers.PQC(self.create_actor_circuit(self.qbits), readout_operators0, differentiator=diff, initializer=tf.keras.initializers.Zeros)(inputs)
        actor = tf.keras.layers.Softmax()(actor)
        a_model = tf.keras.models.Model(inputs=[inputs], outputs=[actor])

        #diff1 = tfq.differentiators.ParameterShift()
        diff1 = tfq.differentiators.SGDifferentiator()
        readout_operators1 = [cirq.Z(self.qbits[3])]
        critic = tfq.layers.PQC(self.create_critic_circuit(self.qbits), readout_operators1, differentiator=diff1, initializer=tf.keras.initializers.Zeros)(inputs)
        c_model = tf.keras.models.Model(inputs=[inputs], outputs=[critic])
        a_model.summary()
        c_model.summary()
        a_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=self._entropy_loss)
        c_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
        return a_model, c_model

    def convert_data(self, data):
        self.qbits = self.make_bits(4)
        ret = []
        data[0] += 4.8
        data[0] *= 2*math.pi/9.6 
        data[1] += 2
        data[1] *= 2*math.pi/4 
        data[2] += 0.48
        data[2] *= 2*math.pi/0.96 
        data[3] += 2
        data[3] *= 2*math.pi/4 
        for i, ang in enumerate(data):
            rx_g = cirq.rx(np.pi*ang)
            ret.append(rx_g(self.qbits[i]))
            rz_g = cirq.rz(np.pi*ang)
            ret.append(rz_g(self.qbits[i]))

        '''
        for i, ang in enumerate(data):
            ang = 0 if ang < 0 else 1
            rx_g = cirq.rx(np.pi*ang)
            ret.append(rx_g(self.qbits[i]))
            rz_g = cirq.rz(np.pi*ang)
            ret.append(rz_g(self.qbits[i]))
        '''
        a = cirq.Circuit()
        a.append(ret)
        inputs = tfq.convert_to_tensor([a])
        return inputs

    def one_qubit_unitary(self, bit, symbols):
        return cirq.Circuit(
            cirq.X(bit)**symbols[0],
            cirq.Y(bit)**symbols[1],
            cirq.Z(bit)**symbols[2])

    def two_qubit_pool(self, source_qubit, sink_qubit, symbols):
        pool_circuit = cirq.Circuit()
        sink_basis_selector = self.one_qubit_unitary(sink_qubit, symbols[0:3])
        source_basis_selector = self.one_qubit_unitary(source_qubit, symbols[3:6])
        pool_circuit.append(sink_basis_selector)
        pool_circuit.append(source_basis_selector)
        pool_circuit.append(cirq.CNOT(control=source_qubit, target=sink_qubit))
        pool_circuit.append(sink_basis_selector**-1)
        return pool_circuit

    def create_actor_circuit(self, bits):
        m = cirq.Circuit()
        m = cirq.Circuit()
        symbols = sympy.symbols('qconv0:252') # 4 qubits * 3 weights per bit * 2 layers
        i = 0
        j = 12
        while j <= 240:
            m += self.layer(symbols[i:j], bits)
            i += 12
            j += 12
        m += self.two_qubit_pool(self.qbits[0], self.qbits[2], symbols[i:i+6])
        m += self.two_qubit_pool(self.qbits[1], self.qbits[3], symbols[i+6:])
        #print(m)
        return m

    def create_critic_circuit(self, bits):
        m = cirq.Circuit()
        symbols = sympy.symbols('critic0:258') # 4 qubits * 3 weights per bit * 2 layers
        i = 0
        j = 12
        while j <= 240:
            m += self.layer(symbols[i:j], bits)
            i += 12
            j += 12
        m += self.two_qubit_pool(self.qbits[0], self.qbits[2], symbols[i:i+6])
        m += self.two_qubit_pool(self.qbits[1], self.qbits[3], symbols[i+6:j])
        m += self.two_qubit_pool(self.qbits[2], self.qbits[3], symbols[j:])
        #print(m)
        return m

    def layer(self, weights, qbits):
        ret = []
        for i in range(len(qbits) - 1):
            ret.append(cirq.CNOT(qbits[i], qbits[i+1]))
        i = 0
        j = 0
        temp = []
        while i < len(qbits):
            rz_g = cirq.rx(weights[j])
            temp.append(rz_g(qbits[i]))
            i += 1
            j += 1
        ret.append(cirq.Moment(temp))
        i = 0
        temp.clear()
        while i < len(qbits):
            ry_g = cirq.ry(weights[j])
            temp.append(ry_g(qbits[i]))
            i += 1
            j += 1
        ret.append(cirq.Moment(temp))
        i = 0
        temp.clear()
        while i < len(qbits):
            rz_g2 = cirq.rz(weights[j])
            ret.append(rz_g2(qbits[i]))
            i += 1
            j += 1
        ret.append(cirq.Moment(temp))
        a = cirq.Circuit()
        a.append(ret)
        return a

    def remember(self, state, action, reward, value):
        self.states.append(state)
        self.rewards.append(reward)
        self.values.append(value)
        self.actions.append(action)

    def get_action(self, obs):
        probs = self.actor.predict(self.convert_data(obs))[0]
        value = self.critic.predict(self.convert_data(obs))[0]
        action = np.random.choice(self.action_space, p=probs)
        return action, value

    def discount_reward(self, rewards):
        d_rewards = np.zeros_like(rewards)
        Gt = 0
        # Discount rewards
        for i in reversed(range(len(rewards))):
            done = 1
            if i == len(rewards) - 1:
                done = 0
                Gt = 0
            else:
                Gt = rewards[i] + self.gamma * self.values[i + 1]
                #Gt = done * Gt * self.gamma + rewards[i]
            d_rewards[i] = Gt

        return d_rewards

    def train(self):
        batch_len = len(self.states)
        state = []
        action = np.zeros(shape=(batch_len, self.action_space))
        
        rewards = self.discount_reward(self.rewards)
        for i in range(batch_len):
            state.append(self.convert_data(self.states[i]))
            action[0][self.actions[i]] = rewards[i] - self.values[i]
        
        self.actor.train_on_batch(state, action)
        self.critic.train_on_batch(state, rewards)
        
        self.states.clear()
        self.rewards.clear()
        self.values.clear()
        self.actions.clear()


# Hyperparameters
ITERATIONS = 1000
windows = 50

env = gym.make("CartPole-v1")
'''env.observation_space.shape'''
print(env.action_space)
print(env.observation_space, env.observation_space.shape)
agent = A2C_agent(env.action_space.n, env.observation_space.shape)
rewards = []
# Uncomment the line before to load model
#agent.q_network = tf.keras.models.load_model("cartpole.h5")
avg_reward = deque(maxlen=ITERATIONS)
best_avg_reward = -math.inf
rs = deque(maxlen=windows)

for i in range(ITERATIONS):
    s1 = env.reset()
    total_reward = 0
    done = False
    while not done:
        #env.render()
        action, value = agent.get_action(s1)
        s2, reward, done, info = env.step(action)
        total_reward += reward
        agent.remember(s1, action, reward, value)
        if done:
            agent.train()
            rewards.append(total_reward)
            rs.append(total_reward)
        else:
            s1 = s2
    if i >= windows:
        avg = np.mean(rs)
        avg_reward.append(avg)
        if avg > best_avg_reward:
            best_avg_reward = avg
            #agent.q_network.save("dqn_cartpole.h5")
    else: 
        avg_reward.append(0)
    
    print("\rEpisode {}/{} || Best average reward {}, Current Iteration Reward {}".format(i, ITERATIONS, best_avg_reward, total_reward) , end='', flush=True)

#np.save("full_q_rew", np.asarray(rewards))
#np.save("full_q_avg", np.asarray(avg_reward))
#plt.ylim(0,300)
plt.plot(rewards, color='olive', label='Reward')
plt.plot(avg_reward, color='red', label='Average')
plt.legend()
plt.ylabel('Reward')
plt.xlabel('Generation')
plt.show()
