import gym
import gym.spaces
import matplotlib.pyplot as plt
import numpy as np
from qlearning_template import QLearningAgent
from sarsa import SarsaAgent
from cliff_walking import CliffWalkingEnv


def play_and_train(env, agent, t_max=10 ** 4):
    """ This function should
    - run a full game (for t_max steps), actions given by agent
    - train agent whenever possible
    - return total reward
    """
    s = env.reset()
    total_reward = 0.0

    for i in range(t_max):
        # Play & train game

        # Update rewards
        a = agent.get_action(s)
        new_s, r, is_done, _ = env.step(a)
        total_reward += r

        # rewards
        agent.update(s, a, new_s, r)
        if is_done:
            break

        # Decay agent epsilon
        # agent.epsilon = ?
        s = new_s
    return total_reward

def getActionRange(state):
    return range(env.nA)


if __name__ == '__main__':
    max_iterations = 100000000000
    visualize = True
    # Create Taxi-v2 env
    # env = gym.make('Taxi-v2')
    env = CliffWalkingEnv()
    env.reset()
    env.render()

    n_states = env.nS
    n_actions = env.nA

    print('States number = %i, Actions number = %i' % (n_states, n_actions))

    # create q learning agent with
    alpha=0.5
    get_legal_actions = lambda s: range(n_actions)
    epsilon=0.2
    discount=0.99

    agent = QLearningAgent(alpha, epsilon, discount, getActionRange)
    agent2 = SarsaAgent(alpha, epsilon, discount, getActionRange)

    plt.figure(figsize=[10, 4])
    rewards1 = []
    rewards2 = []
    # Training loop
    for i in range(max_iterations):
        # Play & train game

        rewards1.append(play_and_train(env, agent))
        rewards2.append(play_and_train(env, agent2))
        if (i + 1) % 100 == 0:
            agent.epsilon = max(agent.epsilon * 0.99, 0.00001)
            agent2.epsilon = max(agent2.epsilon * 0.99, 0.00001)
            # agent.alpha = max(agent.alpha * 0.99, 0.00001)
            # agent2.alpha = max(agent2.alpha * 0.99, 0.00001)

        if i % 100 == 0:
            print('Iteration {}, Average reward {:.2f}, Average reward {:.2f}, Epsilon {:.3f}'.format(i, np.mean(rewards1[-100:]), np.mean(rewards2[-100:]), agent.epsilon))

        if visualize:
            plt.subplot(1, 2, 1)
            plt.plot(rewards1, color='r')
            plt.plot(rewards2, color='b')
            plt.xlabel('Iterations')
            plt.ylabel('Total Reward')

            plt.subplot(1, 2, 2)
            plt.hist(rewards1, bins=20, range=[-700, +20], color='red', label='Rewards distribution')
            plt.hist(rewards2, bins=20, range=[-700, +20], color='blue', label='Rewards distribution')
            plt.xlabel('Reward')
            plt.ylabel('p(Reward)')
            plt.draw()
            plt.pause(0.05)
            plt.cla()
