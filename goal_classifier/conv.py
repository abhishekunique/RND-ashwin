#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

class CNN():
    def __init__(self, learning_rate=0.001, goal_cond=False):
# def cnn_model_fn(features, labels, mode):
        with tf.variable_scope("goal_classifier"):
            """Model function for CNN."""
            # Input Layer
            # Reshape X to 4-D tensor: [batch_size, width, height, channels]
            # MNIST images are 32x32 pixels, and have one color channel
            self.images = tf.placeholder(
                tf.float32,
                shape=(None, 32, 32, 3),
                name='images',
            )

            self.labels = tf.placeholder(tf.int64, shape=(None), name='labels')

            # input_layer = tf.reshape(features["x"], [-1, 32, 32, 3])

            # Convolutional Layer #1
            # Computes 32 features using a 5x5 filter with ReLU activation.
            # Padding is added to preserve width and height.
            # Input Tensor Shape: [batch_size, 32, 32, 1]
            # Output Tensor Shape: [batch_size, 32, 32, 32]
            conv1 = tf.layers.conv2d(
              inputs=self.images,
              filters=32,
              kernel_size=[5, 5],
              padding="same",
              activation=tf.nn.relu)

            # Pooling Layer #1
            # First max pooling layer with a 2x2 filter and stride of 2
            # Input Tensor Shape: [batch_size, 32, 32, 32]
            # Output Tensor Shape: [batch_size, 16, 16, 32]
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

            # Convolutional Layer #2
            # Computes 64 features using a 5x5 filter.
            # Padding is added to preserve width and height.
            # Input Tensor Shape: [batch_size, 16, 16, 32]
            # Output Tensor Shape: [batch_size, 16, 16, 64]
            conv2 = tf.layers.conv2d(
              inputs=pool1,
              filters=64,
              kernel_size=[5, 5],
              padding="same",
              activation=tf.nn.relu)
            # conv2d/bias (DT_FLOAT) [32]
            # conv2d/kernel (DT_FLOAT) [5,5,3,32]
            # conv2d_1/bias (DT_FLOAT) [64]
            # conv2d_1/kernel (DT_FLOAT) [5,5,32,64]
            # dense/bias (DT_FLOAT) [1024]
            # dense/kernel (DT_FLOAT) [4097,1024]
            # dense_1/bias (DT_FLOAT) [2]
            # dense_1/kernel (DT_FLOAT) [1024,2]


            # Pooling Layer #2
            # Second max pooling layer with a 2x2 filter and stride of 2
            # Input Tensor Shape: [batch_size, 16, 16, 64]
            # Output Tensor Shape: [batch_size, 8, 8, 64]
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

            # Flatten tensor into a batch of vectors
            # Input Tensor Shape: [batch_size, 7, 7, 64]
            # Output Tensor Shape: [batch_size, 7 * 7 * 64]
            pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])


            # Dense Layer
            # Densely connected layer with 1024 neurons
            # Input Tensor Shape: [batch_size, 7 * 7 * 64]
            # Output Tensor Shape: [batch_size, 1024]
            if goal_cond:
                self.goals = tf.placeholder(tf.float32, shape=(pool2_flat.shape[0], 1))
                dense = tf.layers.dense(inputs=tf.concat([pool2_flat, self.goals], axis=1), units=1024, activation=tf.nn.relu)
            else:
                dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

            # Add dropout operation; 0.6 probability that element will be kept
            # dropout = tf.layers.dropout(
            #     inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

            # Logits layer
            # Input Tensor Shape: [batch_size, 1024]
            # Output Tensor Shape: [batch_size, 10]
            self.logits = tf.layers.dense(inputs=dense, units=2)
            self.pred_classes = tf.argmax(input=self.logits, axis=1)
            self.pred_probs = tf.nn.softmax(self.logits, name="softmax_tensor")
            self.avg_pred_prob = tf.reduce_mean(tf.reduce_max(self.pred_probs, axis=1))
            correct_pred = tf.equal(self.labels, self.pred_classes)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            # # Calculate Loss (for both TRAIN and EVAL modes)
            self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=self.logits)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            self.train_op = optimizer.minimize(
              loss=self.loss,
              global_step=tf.train.get_global_step())

            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('avg_prob', self.avg_pred_prob)
            tf.summary.scalar('accuracy', self.accuracy)
            self.summary = tf.summary.merge_all()

            # return logits, predictiomerged = tf.summary.merge_all()

            #
            # #
            # # Configure the Training Op (for TRAIN mode)
            # if mode == tf.estimator.ModeKeys.TRAIN:
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            # train_op = optimizer.minimize(
            #     loss=loss,
            #     global_step=tf.train.get_global_step())
            # return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
            #
            # # Add evaluation metrics (for EVAL mode)
            # eval_metric_ops = {
            #   "accuracy": tf.metrics.accuracy(
            #       labels=labels, predictions=predictions["classes"])}
            # return tf.estimator.EstimatorSpec(
            #   mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    def get_variables(self):
        return tf.trainable_variables()


def wrap_dist(theta1, theta2):
    return np.minimum(np.abs(theta1-theta2), 2*np.pi-np.abs(theta1-theta2))


def main(goal_cond):
    import pickle
    import os
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    cur_dir = os.path.dirname(os.path.realpath(__file__))


    if goal_cond:
        exp_file = 'screw_imgs'
    else:
        exp_file = 'goal_neg_images_180_unif'
  # Load training and eval data
    # goal_data = pickle.load(open(cur_dir + '/goal_images_180/goal_images_180.pkl', 'rb'))
    # goal_neg_data = pickle.load(open(cur_dir + '/' + exp_file + '/' + exp_file + '.pkl', 'rb'))
    # data = np.append(goal_data, goal_neg_data, axis=0)
    # train_data, test_data = train_test_split(data, test_size=0.2, shuffle=True)
    # num_train = train_data.shape[0]
    # train_images, train_labels, train_pos = np.array(train_data[:,0].tolist()), np.array(train_data[:,1].tolist()), train_data[:,2]
    # test_images, test_labels, test_pos = np.array(test_data[:,0].tolist()), np.array(test_data[:,1].tolist()), test_data[:,2]

    data = pickle.load(open(cur_dir + '/screw_imgs/screw_imgs.pkl', 'rb'))
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=True)\

    num_train, num_test = train_data.shape[0], test_data.shape[0]
    train_images, train_pos = np.array(train_data[:,0].tolist()), train_data[:,1]
    test_images, test_pos = np.array(test_data[:,0].tolist()), test_data[:,1]


    def batcher(batch_size=100, goal_cond=False):
        i = 0
        while True:
            # If goal conditioned, sample goal, else goal=pi
            if goal_cond:
                batch_goals = np.random.uniform(0,2*np.pi, size=batch_size)
            else:
                batch_goals = np.full(batch_size, fill_value=np.pi)

            # Wraparound logic
            if i + batch_size >= num_train:
                batch_pos = np.append(train_pos[i:], train_pos[:(i+batch_size) % num_train], axis=0)
                if goal_cond:
                    rand_inds = np.random.choice(range(batch_size), size=batch_size // 2, replace=False)
                    batch_goals[rand_inds] = batch_pos[rand_inds]
                batch_labels = (wrap_dist(batch_goals, batch_pos) < 0.15)*1
                batch_images = np.append(train_images[i:], train_images[:(i+batch_size) % num_train], axis=0)
                i = (i+batch_size) % num_train

            # Normal get batch
            else:
                batch_pos = train_pos[i:i + batch_size]

                # If goal_cond, ensure that half of the batch are successes
                if goal_cond:
                    rand_inds = np.random.choice(range(batch_size), size=batch_size // 2, replace=False)
                    batch_goals[rand_inds] = batch_pos[rand_inds]
                batch_labels = (wrap_dist(batch_goals, batch_pos) < 0.15)*1
                batch_images = train_images[i:i + batch_size]
                i += batch_size
            yield batch_images, batch_labels, batch_goals.reshape((batch_size, 1))


    # Create the CNN
    cnn = CNN(goal_cond=goal_cond)

    # Get batch
    train_batch = batcher(batch_size=200, goal_cond=goal_cond)

    # set up tf
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()

    # save data
    train_writer = tf.summary.FileWriter(cur_dir + '/' + exp_file + '/train_scope',
    sess.graph)
    test_writer = tf.summary.FileWriter(cur_dir + '/' + exp_file + '/test_scope')
    pickle.dump(train_data, open(cur_dir + '/' + exp_file + '/train_scope/train_data.pkl', 'wb'))
    pickle.dump(test_data, open(cur_dir + '/' + exp_file + '/test_scope/test_data.pkl', 'wb'))

    # Training
    for i in range(200000):
        # Set up batch
        image_batch, label_batch, goal_batch = train_batch.__next__()
        if image_batch.shape[0] == 0:
            import ipdb; ipdb.set_trace()
        if goal_cond:
            feed_dict = {cnn.images: image_batch, cnn.labels: label_batch, cnn.goals: goal_batch}
        else:
            feed_dict = {cnn.images: image_batch, cnn.labels: label_batch}

        # Train step
        pred_classes, pred_probs, avg_pred_prob, train_acc, loss, summary, _ = sess.run([cnn.pred_classes, cnn.pred_probs, cnn.avg_pred_prob, cnn.accuracy, cnn.loss, cnn.summary, cnn.train_op], feed_dict=feed_dict)
        train_writer.add_summary(summary, i) # print to tensorboard

        # Testing
        if i % 1000 == 0:
            # Test data
            test_goals = np.full((num_test), fill_value=np.pi)
            test_labels = (wrap_dist(test_goals, test_pos) < 0.15) * 1
            if goal_cond:
                test_feed_dict = {cnn.images: test_images, cnn.labels: test_labels, cnn.goals:np.expand_dims(test_goals, 1)}
            else:
                test_feed_dict = {cnn.images: test_images, cnn.labels: test_labels}

            # Evaluation
            test_pred_probs, test_avg_pred_prob, test_acc, test_summary = sess.run([cnn.pred_probs, cnn.avg_pred_prob, cnn.accuracy, cnn.summary], feed_dict=test_feed_dict)
            print("Iter: %i, Train Loss: %f, Avg Pred Prob (Train): %f, Train Acc: %f, Test Acc: %f, Avg Pred Prob (Test): %f" %(i, loss, avg_pred_prob, train_acc, test_acc, test_avg_pred_prob))
            test_writer.add_summary(test_summary, i)

            # Results
            plt.scatter(test_pos, test_pred_probs[:,1])
            plt.xlabel('Angle of Valve (Goal=pi)')
            plt.ylabel('Probability of Being a Goal Image')
            plt.savefig(cur_dir + '/' + exp_file + '/test_scope/' + 'angle_x_prob_%i.png' %i)
            plt.clf()
            print('Graph saved as: ' + exp_file + '/test_scope/' + 'angle_x_prob_%i.png' %i)

            # Save Model
            saver.save(sess, cur_dir + '/' + exp_file + '/train_scope/params.ckpt')
            print("Model saved in path: %s" % exp_file + '/train_scope/params.ckpt')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("simple_example")
    parser.add_argument('--goal_cond', action='store_true', default=False)
    args = parser.parse_args()
    goal_cond = args.goal_cond
    main(goal_cond)
