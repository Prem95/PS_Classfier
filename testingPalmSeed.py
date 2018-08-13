#some basic imports and setups
import os
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

current_dir = './Test_Bank1.txt'
image_dir = os.path.join(current_dir, 'images')
img_path_list=[]

with open(current_dir, 'r') as img_files:

    imgs = []
    lables = []
    for i, f in enumerate (img_files):
        img_path, lable = f.split(' ')
        img_path_list.append(img_path)
        imgs.append(plt.imread(img_path))
        lables.append(lable)
    print("Number of test images = " + str(len(imgs)))

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

# Create a session to import meta graphs
sess = tf.Session()

try:
    saver = tf.train.import_meta_graph('./checkpoint/model_epoch500.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./checkpoint/'))
    saved_dict = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
except RuntimeError as re:
    print("Disable other/eager session() {}", format(re))
    
model = AlexNet(x, keep_prob, 2, [], saved_dict, load_pretrained_weight=True)
model.load_initial_weights(sess)

# Define activation of last layer as score
score = model.fc8

# Calculate softmax 
softmax = tf.nn.softmax(score)

# Iterate over the test images to determine the score 
for index, image in enumerate(imgs):
    
    img = cv2.resize(image.astype(np.float32), (227, 227))
    img = img - imagenet_mean
    img = img.reshape((1, 227, 227, 3))
    classProb = sess.run(softmax, feed_dict={x:img, keep_prob: 1})

    classname = class_names[np.argmax(classProb)]

    with open(resultsFilename, 'a') as file:
        file.write(class_names[6] + " %.3f " %classProb[0, np.argmax(classProb)] + lables[index])

sess.close()

   

