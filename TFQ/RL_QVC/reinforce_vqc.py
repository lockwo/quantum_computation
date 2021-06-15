import gym
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import deque
import tensorflow as tf
from policies import * 

tf.get_logger().setLevel('ERROR')

class REINFORCE(object):
    def __init__(self, a_space, o_space):
        self.action_space = a_space
        self.state_space = o_space
        self.gamma = 0.99
        self.states, self.actions, self.rewards = [], [], []
        self.policy = ReUpPolicy(self.state_space, 5, self.action_space)
        #self.policy = NoReUpPolicy(self.state_space, 5, self.action_space)
        self.opt = tf.keras.optimizers.Adam(lr=0.06)

    def remember(self, s, a, r):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)

    def get_action(self, obs):
        probs = self.policy(np.array([obs])).numpy()[0]
        return np.random.choice(self.action_space, p=probs)

    def discount_rewards(self, rewards):
        d_rewards = np.zeros_like(rewards)
        Gt = 0
        # Discount rewards
        for i in reversed(range(len(rewards))):
            Gt = Gt * self.gamma + rewards[i]
            d_rewards[i] = Gt
        # Normalize
        d_rewards = (d_rewards - np.mean(d_rewards)) / (np.std(d_rewards) + 1e-8)
        return d_rewards

    def update(self):
        state_batch = tf.convert_to_tensor(self.states, dtype=tf.float32)
        action_batch = tf.convert_to_tensor([[i, self.actions[i]] for i in range(len(self.actions))], dtype=tf.int32)
        reward_batch = tf.convert_to_tensor(self.discount_rewards(self.rewards), dtype=tf.float32)

        with tf.GradientTape() as tape:
            model_dist = self.policy(state_batch)
            action_probs = tf.gather_nd(model_dist, action_batch)
            log_probs = tf.math.log(action_probs)
            error = tf.math.reduce_mean(-log_probs * reward_batch)

        grads = tape.gradient(error, self.policy.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.policy.trainable_variables))

        self.states.clear()
        self.actions.clear()
        self.rewards.clear()


if __name__ == "__main__":
    iterations = 300
    rolling_avg = 20

    env = gym.make("CartPole-v1")
    agent = REINFORCE(env.action_space.n, env.observation_space.shape[0])
    rewards = []
    avg_reward = deque(maxlen=iterations)
    best_avg_reward = avg = -math.inf
    rs = deque(maxlen=rolling_avg)
    for i in range(iterations):
        s1 = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.get_action(s1)
            s2, reward, done, _ = env.step(action)
            total_reward += reward
            agent.remember(s1, action, reward)
            s1 = s2
        agent.update()
        rewards.append(total_reward)
        rs.append(total_reward)
        avg = np.mean(rs)
        avg_reward.append(avg)
        if avg > best_avg_reward:
            best_avg_reward = avg
        print("\rEpisode {}/{} || Best average reward {}, Current Avg {}, Current Iteration Reward {}".format(i, iterations, best_avg_reward, avg, total_reward))

    plt.plot(rewards, color='blue', alpha=0.2, label='Reward')
    plt.plot(avg_reward, color='red', label='Average')
    plt.legend()
    plt.ylabel('Reward')
    plt.xlabel('Iteration')
    plt.show()
    plt.savefig("reinforce")
