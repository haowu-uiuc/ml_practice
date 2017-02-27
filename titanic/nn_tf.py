from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import csv
import random

import tensorflow as tf

FLAGS = None


def read_csv_data(file_name):
    with open(file_name, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        data = list()
        skip_first = True
        for row in reader:
            if skip_first:
                skip_first = False
                continue
            data.append(row)
    return data


def extract_label_and_feature(data):
    labels = list()
    selected_features = list()
    for i in range(0, len(data)):

        if data[i][1] == '0':
            label = [0, 1]
        else:
            label = [1, 0]

        allfeature = data[i][2:]
        pclass = int(allfeature[0]) - 1
        sex = 1 if allfeature[2] == 'male' else 0
        age = 0 if allfeature[3] == '' else float(allfeature[3])
        hasage = 0 if allfeature[3] == '' else 1
        sibsp = int(allfeature[4])
        parch = int(allfeature[5])
        fare = 0. if allfeature[7] == '' else float(allfeature[7])
        hasfare = 0 if allfeature[7] == '' else 1
        hasembarked = 0 if allfeature[9] == '' else 1
        embarked_S = 0
        embarked_C = 0
        embarked_Q = 0
        if allfeature[9] == '':
            if (allfeature[9]) == 'S':
                embarked_S = 1
            elif (allfeature[9] == 'C'):
                embarked_C = 1
            elif (allfeature[9] == 'Q'):
                embarked_Q = 1

        selected_feature = [
            pclass,
            sex, age, hasage,
            sibsp, parch, fare, hasfare,
            hasembarked, embarked_S,
            embarked_C, embarked_Q]

        labels.append(label)
        selected_features.append(selected_feature)
    return labels, selected_features


def main(_):
    # Import data
    data = read_csv_data("./train.csv")
    random.shuffle(data)
    train_labels, train_features = extract_label_and_feature(data[:700])
    test_labels, test_features = extract_label_and_feature(data[700:])

    train_labels_batch = list()
    train_features_batch = list()
    batch_num = 10
    batch_size = int(len(train_labels) / batch_num)
    for i in range(0, batch_num):
        if i == batch_num - 1:
            train_labels_batch.append(
                train_labels[(i * batch_size):])
            train_features_batch.append(
                train_features[(i * batch_size):])
        else:
            train_labels_batch.append(
                train_labels[(i * batch_size):((i + 1) * batch_size)])
            train_features_batch.append(
                train_features[(i * batch_size):((i + 1) * batch_size)])


    # Create the model
    x = tf.placeholder(tf.float32, [None, 12])
    W1 = tf.Variable(tf.zeros([12, 12]))
    b1 = tf.Variable(tf.zeros([12]))
    z1 = tf.matmul(x, W1) + b1
    y1 = tf.sigmoid(z1)
    W2 = tf.Variable(tf.zeros([12, 8]))
    b2 = tf.Variable(tf.zeros([8]))
    z2 = tf.matmul(y1, W2) + b2
    y2 = tf.sigmoid(z2)
    # W3 = tf.Variable(tf.zeros([8, 4]))
    # b3 = tf.Variable(tf.zeros([4]))
    # z3 = tf.matmul(y2, W3) + b3
    # y3 = tf.sigmoid(z3)
    W4 = tf.Variable(tf.zeros([8, 2]))
    b4 = tf.Variable(tf.zeros([2]))
    y = tf.matmul(y2, W4) + b4

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 2])

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    train_step = tf.train.GradientDescentOptimizer(
        0.2).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    # init Variables (trainable variables)
    tf.global_variables_initializer().run()
    # Train
    for _ in range(10000):
        for i in range(0, len(train_features_batch)):
            sess.run(
                train_step,
                feed_dict={
                    x: train_features_batch[i],
                    y_: train_labels_batch[i]})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={
        x: test_features, y_: test_labels}))

    # Output result:
    prediction = tf.argmax(y, 1)
    best = sess.run([prediction], feed_dict={
        x: test_features, y_: test_labels})
    print(best)


if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])
