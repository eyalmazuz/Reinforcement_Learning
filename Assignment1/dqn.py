from collections import deque, namedtuple
from datetime import datetime
from itertools import count
import random

import gym
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20, 10)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input

LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 1
EPSILON_START = 0.9
EPSILON_END = 0.1
DECAY_RATE = 200
UPDATE_EVERY = 10

CUR_DATE = str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


class DQN(Model):

    def __init__(self, n_states, n_actions, n_layers):
        super(DQN, self).__init__()

        if n_layers == 3:
            self.dense_layers = [Dense(units=units, activation='relu') for units in [512, 256, 64]]
        
        elif n_layers == 5:
            self.dense_layers = [Dense(units=units, activation='relu') for units in [64, 64, 32, 32, 16]]

        self.action_values = Dense(n_actions, activation='linear')

    def call(self, x):

        for layer in self.dense_layers:
            x = layer(x)

        q_values = self.action_values(x)

        return q_values

class MemoryReplay():

    def __init__(self, size, batch_size):
        
        self.memory = deque([], maxlen=size)
        self.batch_size = batch_size
    
    def sample(self):
        if len(self.memory) < self.batch_size:
            return

        sample = random.sample(self.memory, self.batch_size)
        
        return sample

    def add(self, *args):
        self.memory.append(Transition(*args))

def sample_action(Q_values, epsilon):
    if np.random.rand() < epsilon:
        action = np.random.randint(Q_values.shape[-1])
    else:
        max_actions = np.flatnonzero(Q_values == np.max(Q_values))
        action = np.random.choice(max_actions)

    return action

def q_learning(env, q_network, target_network, learning_rate, 
              discount_factor, epsilon_start, epsilon_end, decay_rate, update_every):
    
    replay_buffer = MemoryReplay(int(1e6), 64)

    rewards = []
    losses = []
    for episode in count():
        state, done = env.reset(), False
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1.0 * episode / decay_rate)
        total_reward = 0
        episode_loss = []
        while not done:
            Q_values = q_network.predict(state.reshape(1, -1))

            action = sample_action(Q_values, epsilon)
            next_state, reward, done, _ = env.step(action)
            
            # env.render()
            total_reward += reward
            replay_buffer.add(state, action, reward, next_state, done)

            state = next_state

        transitions = replay_buffer.sample()
        if not transitions:
            continue

        batch = Transition(*zip(*transitions))
        state_batch = np.array(batch.state)
        action_batch = np.array(batch.action)
        reward_batch = np.array(batch.reward)
        next_state_batch = np.array(batch.next_state)
        done_batch = np.array(batch.done)

        state_q_values = q_network.predict(state_batch)
        next_state_q_values = target_network.predict(next_state_batch)

        for i in range(state_q_values.shape[0]):
            if done_batch[i]:
                state_q_values[i, action_batch[i]] = reward_batch[i]
            else:
               state_q_values[i, action_batch[i]] = reward_batch[i] + discount_factor * next_state_q_values[i].max()
        
        history = q_network.fit(state_batch, state_q_values, batch_size=64, verbose=0)
        loss = history.history['loss'][0]
        episode_loss.append(loss)

        if (episode + 1) % update_every == 0:
            target_network.set_weights(q_network.get_weights())
            
        if episode_loss:
            avg_loss = sum(episode_loss) / len(episode_loss)
            losses.append(avg_loss)
        
        rewards.append(total_reward)
        if losses:
            print(f'Episode: {episode + 1} Loss: {losses[-1]} Reward: {total_reward}')
        
        if sum(rewards[-100:]) / 100 > 475.0:
            break

    return rewards, losses

def main():
    env = gym.make('CartPole-v1')

    q_network = DQN(env.observation_space.shape[0], env.action_space.n, 3)
    target_network = DQN(env.observation_space.shape[0], env.action_space.n, 3)
    
    q_network.compile('Adam', 'MSE')

    target_network.set_weights(q_network.get_weights())

    rewards, losses = q_learning(env, q_network, target_network, LEARNING_RATE, DISCOUNT_FACTOR,
                                EPSILON_START, EPSILON_END, DECAY_RATE, UPDATE_EVERY)

    
    plt.plot(range(len(rewards)), rewards)
    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.title(f'Reward per episode')

    plt.savefig(f'./dqn_rewards_{CUR_DATE}.png', )
    plt.clf()
    
    plt.plot(range(len(losses)), losses)
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.title(f'loss per step')
    plt.savefig(f'./dqn_steps_{CUR_DATE}.png')

if __name__ == "__main__":
    main()
