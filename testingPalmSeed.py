import os
import shutil
import tkinter
import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from alexnetType1 import AlexNet
from caffe_classes import class_names

# Initializer
num_classes = 2
original_class_label = '' # Diff for undiff

imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)

# //* Import the test files
test_path = './Test_Bank1.txt' 
test_path_imagesAppend = os.path.join(test_path, 'images')
testImagePathList = []
testLabelsPathList = []

# //* Open the files using with to auto close wrapper
with open(test_path, 'r') as tp:

    testImages = []
    labels = []

    #! Basic enumerate and split and store in list. Very basic for files searching.
    for index, files in enumerate(tp):

        img, label = files.split(' ') 
        testImagePathList.append(img)
        testLabelsPathList.append(label)
        testImages.append(cv2.imread(img))
        labels.append(label)

print("Total number of files = " + str(len(testImagePathList)))

resultsFilename = './Result_Classifier_1.txt'

# Check file path
if (os.path.isdir('MISCLASSIFIED')) == True:
    print('MISSCLASSIFIED - Exists')
else:
    os.makedirs('MISCLASSIFIED')

if (os.path.isdir('CORRECTCLASSIFIED')) == True:
    print('CORRECTCLASSIFIED - Exists')
else:
    os.makedirs('CORRECTCLASSIFIED')


#! IMPLEMENT THE ALEXNET for classification

x = tf.placeholder(tf.float32, [1, 227, 227, 3])
keep_prob = tf.placeholder(tf.float32)

with tf.Session() as sess:

    saver = tf.train.import_meta_graph('./Checkpoint/model_epoch500.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./Checkpoint/'))
    saved_dict = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    model = AlexNet(x, keep_prob, num_classes, [], saved_dict, load_pretrained_weights=True)
    model.load_initial_weights(sess) # ! Loads the weights from the saver.restore(meta)

# //* Final layer for the score calculation
    score = model.fc8 
    softmax = tf.nn.softmax(score)
    print(softmax)






