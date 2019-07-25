import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
cur_dir = os.path.dirname(os.path.realpath(__file__))
from conv import CNN, wrap_dist
import tensorflow as tf
goal_cond = True
exp_file = 'screw_imgs'
cnn = CNN(goal_cond=goal_cond)

sess = tf.InteractiveSession()
saver = tf.train.Saver()
tf.global_variables_initializer().run()
saver.restore(sess, cur_dir + "/screw_imgs/train_scope/params.ckpt")

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
print_tensors_in_checkpoint_file(cur_dir + "/screw_imgs/train_scope/params.ckpt", all_tensors=False, tensor_name='')

test_data = pickle.load(open(cur_dir + '/screw_imgs_2/screw_imgs.pkl', 'rb'))
num_test = test_data.shape[0]
test_images, test_pos = np.array(test_data[:,0].tolist()), test_data[:,1]

for goal in [0, 0.5, 1, 1.5, 2]:

    test_goals = np.full((num_test, 1), fill_value=goal*np.pi)
    test_labels = (wrap_dist(test_goals, np.expand_dims(test_pos, 1)) < 0.15) * 1
    test_feed_dict = {cnn.images: test_images, cnn.labels: test_labels, cnn.goals:test_goals}
    test_pred_probs, test_avg_pred_prob, test_acc, test_summary = sess.run([cnn.pred_probs, cnn.avg_pred_prob, cnn.accuracy, cnn.summary], feed_dict=test_feed_dict)
    # print("Iter: %i, Train Loss: %f, Avg Pred Prob (Train): %f, Train Acc: %f, Test Acc: %f, Avg Pred Prob (Test): %f" %(i, loss, avg_pred_prob, train_acc, test_acc, test_avg_pred_prob))
    # test_writer.add_summary(test_summary, i)

    plt.scatter(test_pos, test_pred_probs[:,1])
    plt.xlabel('Angle of Valve (Goal={:.2f}_pi)'.format(goal))
    plt.ylabel('Probability of Being a Goal Image')
    plt.savefig((cur_dir + '/' + exp_file + '/test/' + 'test_goal_{:.2f}_pi.png').format(goal))
    plt.clf()
    print(('Graph saved as: ' + exp_file + '/test/' + 'test_goal_{:.2f}_pi.png').format(goal))
