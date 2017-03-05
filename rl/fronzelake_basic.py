import gym
import numpy as np


env = gym.make('FrozenLake-v0')

# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])
N = np.zeros([env.observation_space.n])
# Set learning parameters
lr = .85
y = .99
e = 0.1
num_episodes = 2000
# create lists to contain total rewards and steps per episode
# jList = []
rList = []
best_r = 0
last_100_r = 0
for i in range(num_episodes):
    # Reset environment and get first new observation
    s = env.reset()
    N[s] += 1
    rAll = 0
    d = False
    j = 0
    # The Q-Table learning algorithm
    while j < 99:
        j += 1
        # Choose an action by greedily (with noise) picking from Q table
        a = np.argmax(
            Q[s, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))

        # Get new state and reward from environment
        s1, r, d, _ = env.step(a)
        # Update Q-Table with new knowledge
        Q[s, a] = Q[s, a] + lr * (r + y * np.max(Q[s1, :]) - Q[s, a])

        rAll += r
        s = s1
        if d:
            break
    # jList.append(j)
    rList.append(rAll)

# get the best average score of 100 runs
best_r = 0
for i in range(0, 100):
    best_r += rList[i]
sum_r = best_r
for i in range(100, len(rList)):
    sum_r = sum_r + rList[i] - rList[i - 100]
    best_r = max(best_r, sum_r)

print "Score over time: " + str(sum(rList) / num_episodes) + "\n"
print "Best score = " + str(best_r)
print "Final Q-Table Values"
print Q

# run the game with trained Q table:
j = 0
# env.render()
rAll = 0
for i in range(0, 100):
    s = env.reset()
    while j < 99:
        j += 1
        a = np.argmax(Q[s, :])
        s1, r, d, _ = env.step(a)
        rAll += r
        # env.render()
        s = s1
        if d:
            break

print("average award = " + str(rAll / 100))
