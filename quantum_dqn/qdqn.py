import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import math
from collections import deque
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq 
import sympy

class U_theta(cirq.SingleQubitGate):
    def __init__(self, a00, a01, a10, a11):
        top = math.sqrt(a10**2 + a11**2)
        bot = math.sqrt(a00**2 + a01**2)
        self.theta = math.atan2(top, bot)

    def _unitary_(self):
        return np.array([[math.cos(self.theta), math.sin(self.theta)],
                         [math.sin(self.theta), -math.cos(self.theta)]])

    def _circuit_diagram_info_(self, args):
        return 'U_theta'

class U_a0(cirq.SingleQubitGate):
    def __init__(self, a00, a01, a10, a11):
        self.a00 = a00
        self.a01 = a01
        self.a10 = a10
        self.a11 = a11

    def _unitary_(self):
        return np.array([[self.a00/math.sqrt(self.a00**2 + self.a01**2), self.a01/math.sqrt(self.a00**2 + self.a01**2)],
                         [self.a01/math.sqrt(self.a00**2 + self.a01**2), -self.a00/math.sqrt(self.a00**2 + self.a01**2)]])

    def _circuit_diagram_info_(self, args):
        return 'U_a0'

class U_a1(cirq.SingleQubitGate):
    def __init__(self, a00, a01, a10, a11):
        self.a00 = a00
        self.a01 = a01
        self.a10 = a10
        self.a11 = a11

    def _unitary_(self):
        return np.array([[self.a10/math.sqrt(self.a10**2 + self.a11**2), self.a11/math.sqrt(self.a10**2 + self.a11**2)],
                         [self.a11/math.sqrt(self.a10**2 + self.a11**2), -self.a10/math.sqrt(self.a10**2 + self.a11**2)]])

    def _circuit_diagram_info_(self, args):
        return 'U_a1'

class Quantum_DQN(object):
    def __init__(self, action_size, state_size, batch_size):
        self.action_space = action_size
        self.state_space = state_size
        self.qbits = self.make_bits(4)
        self.q_network = self.make_net(state_size)
        self.memory = deque(maxlen=10000)
        self.batch = batch_size
        # Q Learning Parameters
        self.gamma = 0.95 # DISCOUNT FACTOR, CLOSE TO 1 = LONG TERM
        self.epsilon = 1.0 # Exploration rate
        self.epsilon_decay = 0.9
        self.epsilon_min = 0.01

    def make_bits(self, num):
        bits = []
        for i in range(num):
            bits.append(cirq.GridQubit(0, i))
        return bits

    def make_net(self, state):
        readout_operators = [cirq.Z(self.qbits[i]) for i in range(4)]
        inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
        #differentiator=tfq.differentiators.ParameterShift()
        quantum_model = tfq.layers.PQC(self.create_model_circuit(self.qbits), readout_operators, differentiator=tfq.differentiators.ParameterShift(), initializer=tf.keras.initializers.Zeros)(inputs)
        quantum_model = tf.keras.layers.Dense(self.action_space)(quantum_model)

        q_model = tf.keras.Model(inputs=[inputs], outputs=[quantum_model])
        q_model.summary()
        #print(q_model.get_weights())
        q_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss=tf.keras.losses.MSE)
        return q_model

    def convert_data(self, data):
        self.qbits = self.make_bits(4)
        ret = []
        if len(data) != 4:
            print(data)
        '''
        data[0] = (data[0] + 2.4) * 75
        data[1] = (data[1] + 2) * 90
        data[2] = (data[2] + 41.8) * 4.3
        data[3] = (data[3] + 3) * 60
        for i, ang in enumerate(data):
            ang = math.radians(ang)
            U = cirq.ry(ang)
            ret.append(U(self.qbits[i]))
        '''
        for i, ang in enumerate(data):
            ang = 0 if ang < 0 else 1
            rx_g = cirq.rx(np.pi*ang)
            ret.append(rx_g(self.qbits[i]))
            rz_g = cirq.rz(np.pi*ang)
            ret.append(rz_g(self.qbits[i]))


        a = cirq.Circuit()
        a.append(ret)
        inputs = tfq.convert_to_tensor([a])
        return inputs

    def create_model_circuit(self, bits):
        m = cirq.Circuit()
        symbols = sympy.symbols('qconv0:36') # 4 qubits * 3 weights per bit * 2 layers
        m += self.layer(symbols[:12], bits)
        m += self.layer(symbols[12:24], bits)
        m += self.layer(symbols[24:], bits)
        print(m)
        return m

    def layer(self, weights, qbits):
        ret = []
        for i in range(len(qbits) - 1):
            ret.append(cirq.CNOT(qbits[i], qbits[i+1]))
        i = 0
        j = 0
        temp = []
        while i < len(qbits):
            rz_g = cirq.rz(weights[j])
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

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, obs):
        if random.random() < self.epsilon: 
            return np.random.choice(self.action_space)
        else:
            act = np.argmax(self.q_network.predict(self.convert_data(obs)))
            #print(act, type(act))
            return act

    def train(self):
        minibatch = random.sample(self.memory, self.batch)
        for state, action, reward, next_state, done in minibatch:
            state = self.convert_data(state)
            next_state = self.convert_data(next_state)
            target_f = self.q_network.predict(state)[0]
            if done:
                target_f[action] = reward
            else:
                q_pred = np.amax(self.q_network.predict(next_state)[0])
                target_f[action] = reward + self.gamma*q_pred
            target_f = np.array([target_f,])
            self.q_network.fit(state, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Hyperparameters
ITERATIONS = 1000
batch_size = 32
windows = 50
learn_delay = 1000

env = gym.make("CartPole-v1")
'''env.observation_space.shape'''
print(env.action_space)
print(env.observation_space, env.observation_space.shape)
agent = Quantum_DQN(env.action_space.n, env.observation_space.shape, batch_size)
rewards = []
# Uncomment the line before to load model
#agent.q_network = tf.keras.models.load_model("cartpole.h5")
avg_reward = deque(maxlen=ITERATIONS)
best_avg_reward = avg = -math.inf
rs = deque(maxlen=windows)

for i in range(ITERATIONS):
    s1 = env.reset()
    total_reward = 0
    done = False
    while not done:
        #env.render()
        action = agent.get_action(s1)
        s2, reward, done, info = env.step(action)
        total_reward += reward
        agent.remember(s1, action, reward, s2, done)
        if len(agent.memory) > learn_delay and done:
            agent.train()
        if done:
            rewards.append(total_reward)
            rs.append(total_reward)
        else:
            s1 = s2
    if i >= windows:
        avg = np.mean(rs)
        avg_reward.append(avg)
        if avg > best_avg_reward:
            best_avg_reward = avg
            #agent.q_network.save("cartpole1.h5")
    else: 
        avg_reward.append(0)
    
    print("\rEpisode {}/{} || Best average reward {}, Current Avg {}, Current Iteration Reward {}".format(i, ITERATIONS, best_avg_reward, avg, total_reward), end='', flush=True)

np.save("rewards2", np.asarray(rewards))
np.save("averages2", np.asarray(avg_reward))
plt.ylim(0,510)
plt.plot(rewards, color='olive', label='Reward')
plt.plot(avg_reward, color='red', label='Average')
plt.legend()
plt.ylabel('Reward')
plt.xlabel('Generation')
plt.show()
