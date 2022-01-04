import gym
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import collections


env = gym.make('MountainCar-v0')

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
            self.W11 = tf.get_variable("W11", [self.state_size, 400], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b11 = tf.get_variable("b11", [400], initializer=tf.zeros_initializer())
            self.W21 = tf.get_variable("W21", [400, 300], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b21 = tf.get_variable("b21", [300], initializer=tf.zeros_initializer())
            self.W31 = tf.get_variable("W31", [300, self.action_size], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b31 = tf.get_variable("b31", [self.action_size], initializer=tf.zeros_initializer())

            self.Z11 = tf.add(tf.matmul(self.state, self.W11), self.b11)
            self.A11 = tf.nn.relu(self.Z11)
            
            self.Z21 = tf.add(tf.matmul(self.A11, self.W21), self.b21)
            self.A21 = tf.nn.relu(self.Z21)
            
            self.output_policy = tf.add(tf.matmul(self.A21, self.W31), self.b31)
            
            #State-Value Function Computation
            self.W11_v = tf.get_variable("W11_v", [self.state_size, 400], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b11_v = tf.get_variable("b11_v", [400], initializer=tf.zeros_initializer())
            self.W21_v = tf.get_variable("W21_v", [400, 300], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b21_v = tf.get_variable("b21_v", [300], initializer=tf.zeros_initializer())
            self.W31_v = tf.get_variable("W31_v", [300, 1], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b31_v = tf.get_variable("b31_v", [1], initializer=tf.zeros_initializer())
            
            self.Z11_v = tf.add(tf.matmul(self.state, self.W11_v, name='mul_state_1'), self.b11_v, name='biasdas_state_1')
            self.A11_v = tf.nn.relu(self.Z11_v)
            
            self.Z21_v = tf.add(tf.matmul(self.A11_v, self.W21_v, name='mul_state_1'), self.b21_v, name='biasasd_state_1')
            self.A21_v = tf.nn.relu(self.Z21_v)
            
            self.output_value = tf.add(tf.matmul(self.A21_v, self.W31_v, name='mul_state_2'), self.b31_v, name='biasasd_state_2')

            # Next State-Value computation
            self.Z12_v = tf.add(tf.matmul(self.next_state, self.W11_v, name='mul_next_state_1'), self.b11_v, name='bias_nexasd_state_1')
            self.A12_v = tf.nn.relu(self.Z12_v)
             
            self.Z22_v = tf.add(tf.matmul(self.A12_v, self.W21_v, name='mul_next_state_1'), self.b21_v, name='bias_next_asdstate_1')
            self.A22_v = tf.nn.relu(self.Z22_v)
            
            self.output_next_value = tf.add(tf.matmul(self.A22_v, self.W31_v, name='mul_next_state_2'), self.b31_v, name='bias_asdnext_state_2')
            
            self.output_next_value = tf.multiply(self.output_next_value, tf.subtract(tf.constant(1.0), self.done), name='zero_asdif_done')
            
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
state_size = env.observation_space.shape[0]
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

        