import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import tensorboard
import gym
import numpy as np

ENV_NAME = 'MountainCarContinuous-v0'
INPUT_SHAPE = 6
OUTPUT_SHAPE = 3


class MLP(Model):
    def __init__(self, input_dim , output_dim, hidden_dim, n_layers, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hiddens = [Dense(hidden_dim, activation='relu') for _ in range(n_layers)]

        self.logits = Dense(output_dim, activation='linear') 

    def call(self, x):
        for layer in self.hiddens:
            x = layer(x)
        output = self.logits(x)
        return output 

class ActorCritic():

    def __init__(self, input_dim , output_dim, hidden_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.actor = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dim=32, n_layers=2)
        self.critic = MLP(input_dim=input_dim, output_dim=1, hidden_dim=12, n_layers=1)

    def __call__(self, state):

        output_value = self.critic(state)

        action_probs = self.actor(state)

        return output_value, action_probs



def train(env, model, actor_optimizer, critic_optimizer, n_episodes, gamma, logdir, is_mc=False):
    summary_writer = tf.summary.create_file_writer(f'./{logdir}/')

    rewards = np.zeros((n_episodes, ))
    for episode in range(n_episodes):
        # I = 1
        state, done = env.reset(), False

        while not done:
            with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:

                if state.shape[0] < INPUT_SHAPE:
                    state = np.pad(state, (0, INPUT_SHAPE - state.shape[0]), mode='constant')
                value, actions = model(state[None, ...])

                mask = np.zeros(OUTPUT_SHAPE)
                if is_mc:
                    mask[3:] = -1e9
                else:
                    mask[env.action_space.n:] = -1e9
                action_probs = tf.nn.softmax(tf.squeeze(actions) + mask, axis=-1) 
                action = np.random.choice(np.arange(OUTPUT_SHAPE), p=action_probs.numpy())
                if is_mc:
                    next_state, reward, done, _ = env.step([action - 1])
                else:
                    next_state, reward, done, _ = env.step(action)
                
                rewards[episode] += reward
                
                if next_state.shape[0] < INPUT_SHAPE:
                    next_state = np.pad(next_state, (0, INPUT_SHAPE - next_state.shape[0]), mode='constant')
                next_state_value, _ = model(next_state[None, ...])

                delta = reward + gamma * (1 - done) * next_state_value - value

                actor_loss = - (tf.math.log(action_probs[action]) * tf.stop_gradient(delta)) # * I)
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
        
        with summary_writer.as_default():
            tf.summary.scalar('Reward', rewards[episode], step=episode)
            tf.summary.scalar('Average Reward', average_reward, step=episode)
            tf.summary.scalar('Actor Loss', actor_loss.numpy()[0,0], step=episode)
            tf.summary.scalar('Critic Loss', critic_loss.numpy()[0,0], step=episode)

        print(f'Episode: {episode + 1}, Reward: {rewards[episode]} Average reward: {average_reward}')

        if np.mean(rewards[(episode - 99):episode+1]) > 475.0:
            return


env = gym.make(f'{ENV_NAME}')


model = ActorCritic(INPUT_SHAPE, OUTPUT_SHAPE, 12)

actor_optimizer = Adam(learning_rate=5e-4)
critic_optimizer = Adam(learning_rate=1e-2)

# model.actor.build(input_shape=(None, 4))
train(env, model, actor_optimizer, critic_optimizer, 1000, 1.0, f'{ENV_NAME}', True)
model.actor.save_weights('./weights.h5')

model.actor.load_weights('./weights.h5')