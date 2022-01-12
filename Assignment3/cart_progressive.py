import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils import mask_action, pad_state, write_summary, save_model
from model import ActorCritic, ProgressiveActorCritic
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorboard
import gym
import numpy as np

TRANSFERS = ['Acrobot-v1', 'MountainCarContinuous-v0']
ENV_NAME = 'CartPole-v1'
INPUT_SHAPE = 6
OUTPUT_SHAPE = 3
SAVE_EVERY = 50

def train(env, model, actor_optimizer, critic_optimizer, n_episodes, gamma, logdir):
    summary_writer = tf.summary.create_file_writer(f'./Logs/{logdir}/')

    rewards = np.zeros((n_episodes, ))
    previous_average_reward = -1e5
    for episode in range(n_episodes):
        state, done = env.reset(), False

        while not done:
            with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:

                state = pad_state(state, INPUT_SHAPE)

                value, actions = model(state[None, ...])

                actions = mask_action(actions, OUTPUT_SHAPE, env.action_space.n, is_mc=False)
                action_probs = tf.nn.softmax(actions, axis=-1) 

                action = np.random.choice(np.arange(OUTPUT_SHAPE), p=action_probs.numpy())
                next_state, reward, done, _ = env.step(action)
                
                rewards[episode] += reward
                
                next_state = pad_state(next_state, INPUT_SHAPE)
                next_state_value, _ = model(next_state[None, ...])

                delta = reward + gamma * (1 - done) * next_state_value - value

                actor_loss = - (tf.math.log(action_probs[action]) * tf.stop_gradient(delta))
                critic_loss = delta ** 2

            actor_gradients = actor_tape.gradient(actor_loss, model.actor.trainable_variables + model.actor_adapter.trainable_variables)
            critic_gradients = critic_tape.gradient(critic_loss, model.critic.trainable_variables + model.critic_adapter.trainable_variables)
            
            actor_optimizer.apply_gradients(zip(actor_gradients, model.actor.trainable_variables + model.actor_adapter.trainable_variables))
            critic_optimizer.apply_gradients(zip(critic_gradients, model.critic.trainable_variables + model.critic_adapter.trainable_variables))

            state = next_state

        if episode > 98:
            # Check if solved
            average_reward = np.mean(rewards[(episode - 99):episode+1])
        else:
            average_reward = 0
        
        results = {
            'Reward': rewards[episode],
            'Average Reward': average_reward,
            'Actor Loss': actor_loss.numpy()[0, 0],
            'Critic Loss': critic_loss.numpy()[0, 0]
        }
        write_summary(summary_writer, results, episode)

        if average_reward > previous_average_reward and (episode) % SAVE_EVERY == 0:
            save_model(model.actor, f'./weights/{"_".join(TRANSFERS)}_{ENV_NAME}/actor_episode_{episode}_{average_reward}.h5')
            save_model(model.critic, f'./weights/{"_".join(TRANSFERS)}_{ENV_NAME}/critic_episode_{episode}_{average_reward}.h5')

        previous_average_reward = average_reward

        print(f'Episode: {episode + 1}, Reward: {rewards[episode]} Average reward: {average_reward}')

        if average_reward > 475.0:
            save_model(model.actor, f'./weights/{"_".join(TRANSFERS)}_{ENV_NAME}/actor_episode_{episode}_{average_reward}.h5')
            save_model(model.critic, f'./weights/{"_".join(TRANSFERS)}_{ENV_NAME}/critic_episode_{episode}_{average_reward}.h5')
            return

env = gym.make(f'{ENV_NAME}')

transfer_actors = []
transfer_critics = []
for transfer in TRANSFERS:
    transfer_model = ActorCritic(INPUT_SHAPE, OUTPUT_SHAPE, 12)
    transfer_model.actor.build((None, INPUT_SHAPE))
    transfer_model.critic.build((None, INPUT_SHAPE))

    transfer_model.actor.load_weights(f'./weights/{transfer}/actor.h5')
    transfer_model.critic.load_weights(f'./weights/{transfer}/critic.h5')
    
    transfer_model.actor.trainable = False
    transfer_model.critic.trainable = False

    transfer_actors.append(transfer_model.actor)
    transfer_critics.append(transfer_model.critic)

actor_optimizer = Adam(learning_rate=5e-4)
critic_optimizer = Adam(learning_rate=1e-2)

model = ProgressiveActorCritic(transfer_actors, transfer_critics, INPUT_SHAPE, OUTPUT_SHAPE, 12)

train(env, model, actor_optimizer, critic_optimizer, 1000, 1.0, f'{"_".join(TRANSFERS)}_{ENV_NAME}')