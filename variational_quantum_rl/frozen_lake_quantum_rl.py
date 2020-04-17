import cirq
import tensorflow as tf
import tensorflow_quantum as tfq
from collections import deque
import gym
import numpy as np
import random
import sympy
import math
import colorama
import matplotlib.pyplot as plt


def decimalToBinaryFixLength(_length, _decimal):
	binNum = bin(int(_decimal))[2:]
	outputNum = [int(item) for item in binNum]
	if len(outputNum) < _length:
		outputNum = np.concatenate((np.zeros((_length-len(outputNum),)),np.array(outputNum)))
	else:
		outputNum = np.array(outputNum)
	return outputNum

class memory(object):
    def __init__(self, length):
        self.mem = deque(maxlen=length)

    def remember(self, state, action, reward, next_state, done):
        self.mem.append((state, action, reward, next_state, done))

def prep_angles(angles, qbits):
    i = 0
    ret = []
    for ang in angles:
        rx_g = cirq.rx(np.pi*ang)
        ret.append(rx_g(qbits[i]))
        rz_g = cirq.rz(np.pi*ang)
        ret.append(rz_g(qbits[i]))
        i += 1
    a = cirq.Circuit()
    a.append(ret)
    #print(a)
    return a

def prep_circuit(state, qbits):
    ang = decimalToBinaryFixLength(4, state)
    return tfq.convert_to_tensor([prep_angles(ang, qbits)])

def layer(weights, qbits):
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
    print(a)
    return a

def create_model_circuit(qubits):
    m = cirq.Circuit()
    symbols = sympy.symbols('qconv0:24') # 4 qubite * 3 weights per bit * 2 layers
    m += layer(symbols[:12], qubits)
    m += layer(symbols[12:], qubits)
    print(m)
    return m

def epsilon_greedy(g, actions, model, state):
    if random.random() < g:
        return random.randint(0, actions-1)
    else:
        return np.argmax(model.predict(state)[0])

def train(model, memory, batch_size, gamma, bits):
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        state = prep_circuit(state, bits)
        next_state = prep_circuit(next_state, bits)
        target_f = model.predict(state)[0]
        if done:
            target_f[action] = reward
        else:
            q_pred = model.predict(next_state)[0][action]
            target_f[action] = reward + gamma*q_pred
        target_f = np.array([target_f,])
        model.fit(state, target_f, epochs=1, verbose=0)

bits = []
for i in range(4):
    bits.append(cirq.GridQubit(0, i))
print(bits)
readout_operators = [cirq.Z(i) for i in bits]
'''
state = np.zeros(shape=(4))
inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.float32)
print(inputs)
ang = tfq.layers.AddCircuit()(inputs, prepend=prep_angles(state, bits))
quantum_model = tfq.layers.PQC(create_model_circuit(bits), readout_operators)(ang)
'''

inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
quantum_model = tfq.layers.PQC(create_model_circuit(bits), readout_operators, differentiator=tfq.differentiators.ParameterShift())(inputs)

q_model = tf.keras.Model(inputs=[inputs], outputs=[quantum_model])

q_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.losses.mse)

q_model.summary()

replay_memory = memory(80)

gym.envs.registration.register(
    id='FrozenLake-v1',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery' : False}
)

env = gym.make("FrozenLake-v1")

ITERATIONS = 800
batch_size = 8
windows = 100
explorataion = 1
gamma = 0.95

print(env.action_space)
print(env.observation_space, env.observation_space.shape)
rewards = []
avg_reward = deque(maxlen=ITERATIONS)
best_avg_reward = -math.inf
rs = deque(maxlen=windows)

colorama.init()

for i in range(ITERATIONS):
    s1 = env.reset()
    total_reward = 0
    done = False
    explorataion = explorataion/(explorataion/100 + 1)
    while not done:
        #env.render()
        state1 = prep_circuit(s1, bits)
        action = epsilon_greedy(explorataion, 4, q_model, state1)
        s2, reward, done, info = env.step(action)
        if reward < 1:
            if done:
                reward = -0.2
        total_reward += reward
        replay_memory.remember(s1, action, reward, s2, done)
        #env.render()
        if len(replay_memory.mem) > 16 and done:
            train(q_model, replay_memory.mem, batch_size, gamma, bits)
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
    else: 
        avg_reward.append(-100)
    
    print("\rEpisode {}/{} || Best average reward {}, Current Iteration Reward {}".format(i, ITERATIONS, best_avg_reward, total_reward) , end='', flush=True)

plt.ylim(-0.5,1.5)
plt.plot(rewards, color='olive', label='Reward')
plt.plot(avg_reward, color='red', label='Average')
plt.legend()
plt.ylabel('Reward')
plt.xlabel('Generation')
plt.show()
