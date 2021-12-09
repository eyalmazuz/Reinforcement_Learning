import gym
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import collections


env = gym.make('CartPole-v1')

np.random.seed(1)


class PolicyNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='policy_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):

            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            # Policy Computation
            self.W1 = tf.get_variable("W1", [self.state_size, 12], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b1 = tf.get_variable("b1", [12], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [12, self.action_size], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b2 = tf.get_variable("b2", [self.action_size], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output_policy = tf.add(tf.matmul(self.A1, self.W2), self.b2)
            
            #Value Function Computation
            self.W1_v = tf.get_variable("W1_v", [self.state_size, 12], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b1_v = tf.get_variable("b1_v", [12], initializer=tf.zeros_initializer())
            self.W2_v = tf.get_variable("W2_v", [12, 1], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b2_v = tf.get_variable("b2_v", [1], initializer=tf.zeros_initializer())
            
            self.Z1_v = tf.add(tf.matmul(self.state, self.W1_v), self.b1_v)
            self.A1_v = tf.nn.relu(self.Z1_v)
            self.output_value = tf.add(tf.matmul(self.A1_v, self.W2_v), self.b2_v)
            
            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output_policy))
            
            self.delta = (self.R_t - self.output_value)
            self.value_loss = tf.squeeze(self.delta)
            
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output_policy, labels=self.action)
            self.policy_loss = tf.reduce_mean(self.neg_log_prob * self.delta)
            self.policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.policy_loss)
            
            #Compute loss for value function
            self.value_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.delta)
            


# Define hyperparameters
state_size = 4
action_size = env.action_space.n

max_episodes = 5000
max_steps = 501
discount_factor = 0.99
learning_rate = 0.0004

render = False

# Initialize the policy network
tf.reset_default_graph()
policy = PolicyNetwork(state_size, action_size, learning_rate)


# Start training the agent with REINFORCE algorithm
with tf.Session() as sess:
    
    summary_writer = tf.summary.FileWriter('./baseline_logs/', sess.graph)

    sess.run(tf.global_variables_initializer())
    solved = False
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
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
        episode_transitions = []

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
            episode_transitions.append(Transition(state=state, action=action_one_hot, reward=reward, next_state=next_state, done=done))
            episode_rewards[episode] += reward

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

        # Compute Rt for each time-step t and update the network's weights
        for t, transition in enumerate(episode_transitions):
            total_discounted_return = sum(discount_factor ** i * t.reward for i, t in enumerate(episode_transitions[t:])) # Rt
            feed_dict = {policy.state: transition.state, policy.R_t: total_discounted_return, policy.action: transition.action}
            _, _, loss, policy_summary, value_summary = sess.run([policy.policy_optimizer, policy.value_optimizer,
                                            policy.policy_loss, policy_loss_summary, value_loss_summary], feed_dict)
            summary_writer.add_summary(policy_summary, episode)
            summary_writer.add_summary(value_summary, episode)
