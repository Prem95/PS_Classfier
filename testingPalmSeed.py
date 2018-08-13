#some basic imports and setups
import os
import shutil
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from alexnetType1 import AlexNet
from caffe_classes import class_names
tf.reset_default_graph()

#mean of imagenet dataset in BGR
imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)

current_dir = '/home/stroke95/Desktop/PS_Classfier/Test_Bank1.txt'
image_dir = os.path.join(current_dir, 'images')
img_path_list=[]

with open('/home/stroke95/Desktop/PS_Classfier/Test_Bank1.txt', 'r') as img_files:
    #load all images
    imgs = []
    lables=[]
    for i,f in enumerate (img_files):
        img_path,lable=f.split(' ')
        img_path_list.append(img_path)
        imgs.append(cv2.imread(img_path))
        lables.append(lable)
    print("Number of test images = " + str(len(imgs)))

resultsFilename = './Result_Classifier_1.txt'

# Check file path
if (os.path.isdir('MISCLASSIFIED')) == True:
    print('MISSCLASSIFIED - Exists')
else:
    os.makedirs('MISCLASSIFIED')

if (os.path.isdir('CORRECTCLASSIFIED')) == True:
    print('CORRECTCLASSIFIED - Exists')

#placeholder for input and dropout rate

x = tf.placeholder(tf.float32, [1, 227, 227, 3])
keep_prob = tf.placeholder(tf.float32)


sess = tf.Session()
saver = tf.train.import_meta_graph('/home/stroke95/Desktop/PS_Classfier/checkpoint/model_epoch500.ckpt.meta')
saver.restore(sess, tf.train.latest_checkpoint('/home/stroke95/Desktop/PS_Classfier/checkpoint/'))
saved_dict = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
model = AlexNet(x, keep_prob, 2, [], saved_dict, load_pretrained_weight=True)
model.load_initial_weights(sess)

# Define activation of last layer as score
score = model.fc8

# Create op to calculate softmax 
softmax = tf.nn.softmax(score)

