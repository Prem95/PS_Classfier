import numpy as np 
import tensorflow as tf 
import os
import cv2
from matplotlib import pyplot as plt
import shutil
from alexnetType1 import AlexNet
from caffe_classes import class_names

# Initializer
num_classes = 2
original_class_label = ''

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

resultsFilename = './TestingResults/Result_Classifier_1.txt'

# Check file path
if (os.path.isdir('MISCLASSIFIED')) == True:
    print('MISSCLASSIFIED -  Exists')
else:
    os.makedirs('MISCLASSIFIED')

if (os.path.isdir('CORRECTCLASSIFIED')) == True:
    print('CORRECTCLASSIFIED - Exists')
else:
    os.makedirs('CORRECTCLASSIFIED')








    
    