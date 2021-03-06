import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

TRAIN_FLAG = False

env = gym.make('FrozenLake-v0')

tf.reset_default_graph()
# These lines establish the feed-forward part of
# the network used to choose actions
inputs1 = tf.placeholder(shape=[1, 16], dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16, 4], 0, 0.01), name='W')
Qout = tf.matmul(inputs1, W)
predict = tf.argmax(Qout, 1)

# Below we obtain the loss by taking the sum of
# squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1, 4], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)


# Initialize table with all zeros
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()

if TRAIN_FLAG:
    # Set learning parameters
    y = .99
    e = 0.1
    num_episodes = 2000
    # create lists to contain total rewards and steps per episode
    jList = []
    rList = []

    for i in range(num_episodes):
        # Reset environment and get first new observation
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        # The Q-Table learning algorithm
        while j < 99:
            j += 1
            # Choose an action by greedily (with e chance of random action)
            # from the Q-network
            a, allQ = sess.run(
                [predict, Qout],
                feed_dict={inputs1: np.identity(16)[s:s + 1]})
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()
            # Get new state and reward from environment
            s1, r, d, _ = env.step(a[0])
            # Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(
                Qout, feed_dict={inputs1: np.identity(16)[s1:s1 + 1]})
            # Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0, a[0]] = r + y * maxQ1
            # Train our network using target and predicted Q values
            _, W1 = sess.run(
                [updateModel, W],
                feed_dict={inputs1: np.identity(16)[s:s + 1], nextQ: targetQ})
            rAll += r
            s = s1
            if d:
                # Reduce chance of random action as we train the model.
                e = 1. / ((i / 50) + 10)
                break
        jList.append(j)
        rList.append(rAll)
        if i % 100 == 0:
            print "Percent of succesful episodes: "\
                + str(sum(rList) / (i + 1) * 100) + "%"

    # save the model
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    saver.save(sess, './checkpoints/frozenlake')

    # get the best average score of 100 runs
    best_r = 0
    for i in range(0, 100):
        best_r += rList[i]
    sum_r = best_r
    for i in range(100, len(rList)):
        sum_r = sum_r + rList[i] - rList[i - 100]
        best_r = max(best_r, sum_r)

    print "Score over time: " + str(sum(rList) / num_episodes) + "\n"
    print "Best average 100-run score: " + str(best_r / 100.)
    print "Final Q-Table Values"

    print "W = " + str(sess.run(W))

    # plt.plot(rList)
    # plt.plot(jList)

else:
    # run the game with trained Q table:
    # new_saver = tf.train.import_meta_graph('./checkpoints/frozenlake.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./checkpoints'))
    print "W = " + str(sess.run(W))

    j = 0
    s = env.reset()
    # env.render()
    rAll = 0.
    num_episode = 100
    for i in range(0, num_episode):
        while j < 99:
            j += 1
            a = sess.run(
                predict, feed_dict={inputs1: np.identity(16)[s:s + 1]})
            s1, r, d, _ = env.step(a[0])
            rAll += r
            # env.render()
            s = s1
            if d:
                break

    print("average award = " + str(rAll / num_episode))
