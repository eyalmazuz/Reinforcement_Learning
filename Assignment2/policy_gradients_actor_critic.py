import gym
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import collections


env = gym.make('CartPole-v1')

np.random.seed(1)


class ActorCritic:
    def __init__(self, state_size, action_size, actor_learning_rate, critic_learning_rate, discount_factor, name='Actric'):
        self.state_size = state_size
        self.action_size = action_size
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.discount_factor = discount_factor

        with tf.variable_scope(name):

            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.placeholder(tf.float32, name="reward")
            self.next_state = tf.placeholder(tf.float32, [None, self.state_size], name="next_state")
            self.done = tf.placeholder(tf.float32, name='done')
            self.I = tf.placeholder(tf.float32, name="I")

            # Policy Computation
            self.W1 = tf.get_variable("W1", [self.state_size, 12], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b1 = tf.get_variable("b1", [12], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [12, self.action_size], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b2 = tf.get_variable("b2", [self.action_size], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output_policy = tf.add(tf.matmul(self.A1, self.W2), self.b2)
            
            #State-Value Function Computation
            self.W1_v = tf.get_variable("W1_v", [self.state_size, 32], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b1_v = tf.get_variable("b1_v", [32], initializer=tf.zeros_initializer())
            self.W2_v = tf.get_variable("W2_v", [32, 1], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b2_v = tf.get_variable("b2_v", [1], initializer=tf.zeros_initializer())
            
            self.Z1_v = tf.add(tf.matmul(self.state, self.W1_v, name='mul_state_1'), self.b1_v, name='bias_state_1')
            self.A1_v = tf.nn.relu(self.Z1_v)
            self.output_value = tf.add(tf.matmul(self.A1_v, self.W2_v, name='mul_state_2'), self.b2_v, name='bias_state_2')

            # Next State-Value computation
            self.Z2_v = tf.add(tf.matmul(self.next_state, self.W1_v, name='mul_next_state_1'), self.b1_v, name='bias_next_state_1')
            self.A2_v = tf.nn.relu(self.Z2_v)
            self.output_next_value = tf.add(tf.matmul(self.A2_v, self.W2_v, name='mul_next_state_2'), self.b2_v, name='bias_next_state_2')
            
            self.output_next_value = tf.multiply(self.output_next_value, tf.subtract(tf.constant(1.0), self.done), name='zero_if_done')
            
            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output_policy))
            
            self.delta = tf.add(self.R_t, tf.constant(self.discount_factor) * self.output_next_value - self.output_value, name='delta')
            
            self.value_loss = tf.squeeze(self.delta)
            
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output_policy, labels=self.action)
            self.policy_loss = tf.reduce_mean(self.neg_log_prob * self.I * tf.stop_gradient(self.delta))
            self.policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.actor_learning_rate).minimize(self.policy_loss)
            
            #Compute loss for value function
            self.value_optimizer = tf.train.AdamOptimizer(learning_rate=self.critic_learning_rate).minimize(self.I * tf.stop_gradient(self.delta) * self.output_value)
            


# Define hyperparameters
state_size = 4
action_size = env.action_space.n

max_episodes = 5000
max_steps = 501
discount_factor = 1.0
actor_learning_rate = 3e-4
critic_learning_rate = 5e-4

render = False

# Initialize the policy network
tf.reset_default_graph()
policy = ActorCritic(state_size, action_size, actor_learning_rate, critic_learning_rate, discount_factor)


# Start training the agent with REINFORCE algorithm
with tf.Session() as sess:
    
    summary_writer = tf.summary.FileWriter('./actorcritic_logs/', sess.graph)

    sess.run(tf.global_variables_initializer())
    solved = False
    episode_rewards = np.zeros(max_episodes)
    average_rewards = 0.0
    
    steps = 0

    
    reward_tensor = tf.placeholder(tf.float32, name="reward_tensor")
    reward_summary = tf.summary.scalar(name='reward', tensor=reward_tensor)
    
    average_reward_tensor = tf.placeholder(tf.float32, name="average_reward_tensor")
    average_reward_summary = tf.summary.scalar(name='average reward', tensor=average_reward_tensor)
    
    policy_loss_summary = tf.summary.scalar(name='policy loss', tensor=policy.policy_loss)
    value_loss_summary = tf.summary.scalar(name='value loss', tensor=policy.value_loss)
    
    for episode in range(max_episodes):
        state = env.reset()
        state = state.reshape([1, state_size])
        I = 1.0
        for step in range(max_steps):
            steps += 1
            actions_distribution = sess.run(policy.actions_distribution, {policy.state: state})
            action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape([1, state_size])

            if render:
                env.render()

            action_one_hot = np.zeros(action_size)
            action_one_hot[action] = 1

            feed_dict = {policy.state: state, policy.R_t: reward, policy.done: done,
                         policy.action: action_one_hot, policy.next_state: next_state, policy.I: I}
            _, _, loss, policy_summary, value_summary = sess.run([policy.policy_optimizer, policy.value_optimizer,
                                            policy.policy_loss, policy_loss_summary, value_loss_summary], feed_dict)

            I = discount_factor * I
            action_one_hot = np.zeros(action_size)
            action_one_hot[action] = 1
            episode_rewards[episode] += reward

            summary_writer.add_summary(policy_summary, episode)
            summary_writer.add_summary(value_summary, episode)

            if done:
                if episode > 98:
                    # Check if solved
                    average_rewards = np.mean(episode_rewards[(episode - 99):episode+1])
                    
                    # writes the average reward 
                    summary = sess.run(average_reward_summary, {average_reward_tensor: average_rewards})
                    summary_writer.add_summary(summary, episode)
                
                print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode], round(average_rewards, 2)))
                
                # writes the current episode reward
                summary = sess.run(reward_summary, {reward_tensor: episode_rewards[episode]})
                summary_writer.add_summary(summary, episode)
                
                if average_rewards > 475:
                    print(' Solved at episode: ' + str(episode))
                    solved = True
                break
            state = next_state

        if solved:
            break

        
