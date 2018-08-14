import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from alexnetType1 import AlexNet
from caffe_classes import class_names
import shutil
num_classes = 2
original_class_lable = ''

imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)

current_dir = './Test_Bank1.txt'
image_dir = os.path.join(current_dir, 'images')
img_path_list = []

with open(current_dir, 'r') as img_files:

    imgs = []
    test_labels = []
    for i, f in enumerate (img_files):
        img_path, lable = f.split(' ')
        img_path_list.append(img_path)
        imgs.append(plt.imread(img_path))
        test_labels.append(lable)

resultsFilename = './Result_Classifier_Bank1.txt'

# Check file path
if (os.path.isdir('MISCLASSIFIED')) == True:
    print('MISSCLASSIFIED - Exists')
else:
    os.makedirs('MISCLASSIFIED')

if (os.path.isdir('CORRECTCLASSIFIED')) == True:
    print('CORRECTCLASSIFIED - Exists')


# Placeholder for input and dropout rate
x = tf.placeholder(tf.float32, [1, 227, 227, 3])
keep_prob = tf.placeholder(tf.float32)

with tf.Session() as sess:

    saver = tf.train.import_meta_graph('./checkpoint/model_epoch500.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./checkpoint/'))
    saved_dict = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    model = AlexNet(x, keep_prob, num_classes, [], saved_dict, load_pretrained_weight=True)
    model.load_initial_weights(sess)
    score = model.fc8  
    softmax = tf.nn.softmax(score)

    fig2 = plt.figure(figsize = (10, 5))

    for i, image in enumerate(imgs):

        img = cv2.resize(image.astype(np.float32), (227,227))
        img = img - imagenet_mean
        img = img.reshape((1, 227, 227, 3))
        probs = sess.run(softmax, feed_dict={x: img, keep_prob: 1})
        class_name = class_names[np.argmax(probs)]
        original_class_lable = class_name[5]
        print("Predicted:" + class_name + " Actual Class: " + test_labels[i] + "Probability: %.4f" %probs[0, np.argmax(probs)])



