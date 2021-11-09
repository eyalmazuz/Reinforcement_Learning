import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

plt.rcParams["figure.figsize"] = (20, 10)
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 1
EPSILON_START = 0.9
EPSILON_END = 0.1
DECAY_RATE = 200

def sample_action(Q_values, epsilon):
    if np.random.rand() < epsilon:
        action = np.random.randint(Q_values.shape[0])
    else:
        max_actions = np.flatnonzero(Q_values == np.max(Q_values))
        action = np.random.choice(max_actions)

    return action

def q_learning(env, learning_rate, discount_factor, epsilon_start, epsilon_end, decay_rate):
    rewards = []
    steps = []
    Q = np.zeros((env.nS, env.nA))
    for episode in trange(5000):
        total_reward = 0
        state, done = env.reset(), False
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1.0 * episode / decay_rate)
        for step in range(100):
            action = sample_action(Q[state], epsilon)
            next_state, reward, done, _ = env.step(action)

            total_reward += reward
            if done:
                target = reward
            else:
                target = reward + discount_factor * Q[next_state].max()
            Q[state, action] = (1 - learning_rate) * Q[state, action] + learning_rate * target
            state = next_state
            if done:
                break

        steps.append(step)
        rewards.append(total_reward)

        if episode in [499, 1999, 4999]:
            plt.imshow(Q, cmap='hot')
            plt.xlabel('actions')
            plt.ylabel('states')
            plt.title(f'Q table Episode {episode + 1}')
            plt.savefig(f'Q_table_Episode_{episode + 1}.png', transparent=True)
            plt.clf()
            print(f'Episode: {episode}\n', Q)

    return Q, rewards, steps

def main():
    env = gym.make('FrozenLake-v0')

    Q, rewards, steps = q_learning(env, LEARNING_RATE, DISCOUNT_FACTOR, EPSILON_START, EPSILON_END, DECAY_RATE)

    
    plt.plot(range(5000), rewards)
    plt.xlabel('episodes')
    plt.ylabel('reward')
    plt.title(f'Reward per episode')

    plt.savefig('./rewards.png', )
    plt.clf()
    
    plt.plot(range(50), np.array(steps).reshape(-1, 100).mean(axis=1))
    plt.xlabel('episodes')
    plt.ylabel('steps')
    plt.title(f'average steps per 100 episodes')
    plt.savefig('./steps.png')

    print(Q.argmax(1).reshape(4, 4))
    print(sum(rewards))

if __name__ == "__main__":
    main()
    ["SFFF"
     "FHFH"
     "FFFH"
     "HFFG"],