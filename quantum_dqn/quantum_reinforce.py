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


def _entropy_loss(target, output):
    return -tf.math.reduce_mean(target*tf.math.log(output))

class REINFORCE_agent(object):
    def __init__(self, action_size, state_size):
        self.action_space = action_size
        self.state_space = state_size
        self.qbits = self.make_bits(4)
        self.policy_net = self.make_net(self.state_space)
        self.gamma = 0.99 # DISCOUNT FACTOR, CLOSE TO 1 = LONG TERM
        self.states, self.actions, self.rewards = [], [], []

    def make_bits(self, num):
        bits = []
        for i in range(4):
            bits.append(cirq.GridQubit(0, i))
        return bits

    def make_net(self, state):
        readout_operators = [cirq.Z(self.qbits[i]) for i in range(4)]
        inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
        #differentiator=tfq.differentiators.ParameterShift()
        quantum_model = tfq.layers.PQC(self.create_model_circuit(self.qbits), readout_operators, differentiator=tfq.differentiators.ParameterShift(), initializer=tf.keras.initializers.Zeros)(inputs)
        #quantum_model = tf.keras.layers.Softmax()(quantum_model)
        quantum_model = tf.keras.layers.Dense(self.action_space, activation='softmax')(quantum_model)

        q_model = tf.keras.Model(inputs=[inputs], outputs=[quantum_model])
        q_model.summary()
        q_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss=_entropy_loss)
        return q_model

    def convert_data(self, data):
        self.qbits = self.make_bits(4)
        ret = []
        if len(data) != 4:
            print(data)
        #data[0] = (data[0] + 2.4) * 75
        #data[1] = (data[1] + 2) * 90
        #data[2] = (data[2] + 41.8) * 4.3
        #data[3] = (data[3] + 3) * 60
        #for i, ang in enumerate(data):
            #ang = math.radians(ang)
            #U = cirq.ry(ang)
            #ret.append(U(self.qbits[i]))
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
        ret.append(cirq.CNOT(qbits[0], qbits[1]))
        ret.append(cirq.CNOT(qbits[1], qbits[2]))
        ret.append(cirq.CNOT(qbits[2], qbits[3]))
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

    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def get_action(self, obs):
        probs = self.policy_net.predict(np.array(self.convert_data(obs)))[0]
        action = np.random.choice(self.action_space, p=probs)
        return action

    def discount_reward(self, rewards):
        d_rewards = np.zeros_like(rewards)
        Gt = 0
        # Discount rewards
        for i in reversed(range(len(rewards))):
            Gt = Gt * self.gamma + rewards[i]
            d_rewards[i] = Gt
        # Normalize
        d_rewards -= np.mean(d_rewards)
        d_rewards /= np.std(d_rewards) 
        return d_rewards

    def train(self):
        batch_len = len(self.states)

        rewards = self.discount_reward(self.rewards)


        action = np.zeros(shape=(1, self.action_space))
        for i in range(batch_len):
            state = self.convert_data(self.states[i])
            action[0][self.actions[i]] = rewards[i]
            self.policy_net.fit(state, action, epochs=1, verbose=0)

        self.states.clear()
        self.actions.clear()    
        self.rewards.clear()



# Hyperparameters
ITERATIONS = 250
windows = 50

env = gym.make("CartPole-v1")
#env.observation_space.shape
print(env.action_space)
print(env.observation_space, env.observation_space.shape[0])
agent = REINFORCE_agent(env.action_space.n, env.observation_space.shape[0])
rewards = []
# Uncomment the line before to load model
#agent.q_network = tf.keras.models.load_model("reinforce_cartpole.h5")
avg_reward = deque(maxlen=ITERATIONS)
best_avg_reward = -math.inf
rs = deque(maxlen=windows)

for i in range(ITERATIONS):
    done = False
    s1 = env.reset()
    total_reward = 0
    while not done:
        #env.render()
        action = agent.get_action(s1)
        s2, reward, done, info = env.step(action)
        total_reward += reward
        agent.remember(s1, action, reward)
        s1 = s2
        
    agent.train()
    rewards.append(total_reward)
    rs.append(total_reward)
    if i >= windows:
        avg = np.mean(rs)
        avg_reward.append(avg)
        if avg > best_avg_reward:
            best_avg_reward = avg
            #agent.policy_net.save("reinforce_cartpole.h5")
    else: 
        avg_reward.append(0)
    
    print("\rEpisode {}/{} || Best average reward {}, Current Iteration Reward {}".format(i, ITERATIONS, best_avg_reward, total_reward), end='', flush=True)

np.save("rewards", np.asarray(rewards))
np.save("averages", np.asarray(avg_reward))
plt.ylim(0,500)
plt.plot(rewards, color='olive', label='Reward')
plt.plot(avg_reward, color='red', label='Average')
plt.legend()
plt.ylabel('Reward')
plt.xlabel('Generation')
plt.show()
