from numpy.lib.function_base import gradient
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import tensorboard
import gym
import numpy as np


class MLP(Model):
    def __init__(self, input_dim , output_dim, hidden_dim, activation, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hidden = Dense(hidden_dim, activation='relu')

        self.logits = Dense(output_dim, activation=activation) 

    def call(self, state):
        hidden = self.hidden(state)
        output = self.logits(hidden)
        return output 

class ActorCritic():

    def __init__(self, input_dim , output_dim, hidden_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.actor = MLP(input_dim, output_dim, 32, 'softmax')
        self.critic = MLP(input_dim, 1, 12, None)

    def __call__(self, state):

        output_value = self.critic(state)

        action_probs = self.actor(state)

        return output_value, action_probs



def train(env, model, actor_optimizer, critic_optimizer, n_episodes, gamma, logdir):

    rewards = np.zeros((n_episodes, ))
    for episode in range(n_episodes):
        # I = 1
        state, done = env.reset(), False

        while not done:
            with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:

                value, action_probs = model(state[None, ...])

                action = np.random.choice(np.arange(model.output_dim), p=action_probs.numpy().ravel())
                next_state, reward, done, _ = env.step(action)
                
                rewards[episode] += reward
                
                next_state_value, _ = model(next_state[None, ...])

                delta = reward + gamma * (1 - done) * next_state_value - value

                actor_loss = - (tf.math.log(action_probs[:, action]) * tf.stop_gradient(delta)) # * I)
                critic_loss = delta ** 2 # * I

            actor_gradients = actor_tape.gradient(actor_loss, model.actor.trainable_variables)
            critic_gradients = critic_tape.gradient(critic_loss, model.critic.trainable_variables)
            
            actor_optimizer.apply_gradients(zip(actor_gradients, model.actor.trainable_variables))
            critic_optimizer.apply_gradients(zip(critic_gradients, model.critic.trainable_variables))

            # I = I * gamma
            state = next_state

        if episode > 98:
            # Check if solved
            average_reward = np.mean(rewards[(episode - 99):episode+1])
        else:
            average_reward = 0
        
        print(f'Episode: {episode + 1}, Reward: {rewards[episode]} Average reward: {average_reward}')

        if np.mean(rewards[(episode - 99):episode+1]) > 475.0:
            return

env = gym.make('CartPole-v1')


model = ActorCritic(env.observation_space.shape[0], env.action_space.n, 12)

actor_optimizer = Adam(learning_rate=5e-4)
critic_optimizer = Adam(learning_rate=1e-2)

train(env, model, actor_optimizer, critic_optimizer, 1000, 1.0, '')