from collections import deque, namedtuple
from itertools import count
import random

import gym
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import gradient
from tqdm import trange
plt.rcParams["figure.figsize"] = (20, 10)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 1
EPSILON_START = 0.9
EPSILON_END = 0.1
DECAY_RATE = 200
UPDATE_EVERY = 10

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])


class DQN(Model):

    def __init__(self, n_states, n_actions, n_layers):
        super(DQN, self).__init__()

        if n_layers == 3:
            self.dense_layers = [Dense(units=units, activation='relu') for units in [64, 64, 16]]
        
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

def build_model(n_states, n_actions, n_layers):
    inputs = Input(shape=(4, ))

    x = inputs
    for _ in range(n_layers):
        x = Dense(64, activation='relu')(x)

    output = Dense(n_actions, activation=None)(x)

    model = Model(inputs=inputs, outputs=output)
    model.compile('SGD', 'MSE',)

    return model

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

        # state_q_values[np.arange(state_q_values.shape[0]), action_batch] = reward_batch + (np.logical_not(done_batch)) * np.amax(next_state_action_values, axis=-1)

        # next_state_action_values = next_state_q_values.max(axis=-1)
        # action_values_rewards = np.zeros(state_batch.shape[0])
        # action_values_rewards += reward_batch
        # action_values_rewards += discount_factor * (np.logical_not(done_batch)) * next_state_action_values
        
        # state_action_values = q_network.predict(state_batch)
        # state_action_values[np.arange(state_action_values.shape[0]), action_batch] = action_values_rewards
        
        history = q_network.fit(state_batch, state_q_values, batch_size=64, verbose=0)
        loss = history.history['loss'][0]
        losses.append(loss)

        if (episode + 1) % update_every == 0:
            target_network.set_weights(q_network.get_weights())

        rewards.append(total_reward)
        if losses:
            print(f'Episode: {episode + 1} Loss: {losses[-1]} Reward: {total_reward}')
        
        

        if sum(rewards[-100:]) / 100 > 475.0:
            break

    return rewards

def main():
    env = gym.make('CartPole-v1')

    q_network = DQN(env.observation_space.shape[0], env.action_space.n, 3)
    target_network = DQN(env.observation_space.shape[0], env.action_space.n, 3)

    q_network.compile('Adam', 'MSE')

    target_network.set_weights(q_network.get_weights())

    rewards = q_learning(env, q_network, target_network, LEARNING_RATE, DISCOUNT_FACTOR,
                                EPSILON_START, EPSILON_END, DECAY_RATE, UPDATE_EVERY)

    
    plt.plot(range(5000), rewards)
    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.title(f'Reward per episode')

    plt.savefig('./rewards.png', )
    plt.clf()
    
    # plt.plot(range(50), np.array(steps).reshape(-1, 100).mean(axis=1))
    # plt.xlabel('episodes')
    # plt.ylabel('steps')
    # plt.title(f'average steps per 100 episodes')
    # plt.savefig('./steps.png')

if __name__ == "__main__":
    main()
