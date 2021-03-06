import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import numpy as np
import cv2
from sklearn.utils import shuffle

loadedImages = []

#Load Images From A
for i in range(0, 1000):
    image = cv2.imread('Dataset/AImages/signA_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))

#Load Images From B
for i in range(0, 1000):
    image = cv2.imread('Dataset/BImages/signB_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))

#Load Images From C
for i in range(0, 1000):
    image = cv2.imread('Dataset/CImages/signC_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))

#Load Images From D
for i in range(0, 1000):
    image = cv2.imread('Dataset/DImages/signD_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))

#Load Images From E
for i in range(0, 1000):
    image = cv2.imread('Dataset/EImages/signE_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))

#Load Images From F
for i in range(0, 1000):
    image = cv2.imread('Dataset/FImages/signF_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))

#Load Images From G
for i in range(0, 1000):
    image = cv2.imread('Dataset/GImages/signG_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))
    
#Load Images From H
for i in range(0, 1000):
    image = cv2.imread('Dataset/HImages/signH_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))

#Load Images From I
for i in range(0, 1000):
    image = cv2.imread('Dataset/IImages/signI_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))

# Create OutputVector

outputVectors = []
for i in range(0, 1000):
    outputVectors.append([1, 0, 0, 0, 0, 0, 0, 0, 0])

for i in range(0, 1000):
    outputVectors.append([0, 1, 0, 0, 0, 0, 0, 0, 0])

for i in range(0, 1000):
    outputVectors.append([0, 0, 1, 0, 0, 0, 0, 0, 0])

for i in range(0, 1000):
    outputVectors.append([0, 0, 0, 1, 0, 0, 0, 0, 0])

for i in range(0, 1000):
    outputVectors.append([0, 0, 0, 0, 1, 0, 0, 0, 0])

for i in range(0, 1000):
    outputVectors.append([0, 0, 0, 0, 0, 1, 0, 0, 0])

for i in range(0, 1000):
    outputVectors.append([0, 0, 0, 0, 0, 0, 1, 0, 0])

for i in range(0, 1000):
    outputVectors.append([0, 0, 0, 0, 0, 0, 0, 1, 0])

for i in range(0, 1000):
    outputVectors.append([0, 0, 0, 0, 0, 0, 0, 0, 1])


testImages = []

#Load Images From A
for i in range(0, 100):
    image = cv2.imread('Dataset/AImages/signA_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))

#Load Images From B
for i in range(0, 100):
    image = cv2.imread('Dataset/BImages/signB_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))

#Load Images From C
for i in range(0, 100):
    image = cv2.imread('Dataset/CImages/signC_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))

#Load Images From D
for i in range(0, 100):
    image = cv2.imread('Dataset/DImages/signD_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))

#Load Images From E
for i in range(0, 100):
    image = cv2.imread('Dataset/EImages/signE_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))

#Load Images From F
for i in range(0, 100):
    image = cv2.imread('Dataset/FImages/signF_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))

#Load Images From G
for i in range(0, 100):
    image = cv2.imread('Dataset/GImages/signG_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))
    
#Load Images From H
for i in range(0, 100):
    image = cv2.imread('Dataset/HImages/signH_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))

#Load Images From I
for i in range(0, 100):
    image = cv2.imread('Dataset/IImages/signI_' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))

testLabels = []

for i in range(0, 100):
    testLabels.append([1, 0, 0, 0, 0, 0, 0, 0, 0])

for i in range(0, 100):
    testLabels.append([0, 1, 0, 0, 0, 0, 0, 0, 0])

for i in range(0, 100):
    testLabels.append([0, 0, 1, 0, 0, 0, 0, 0, 0])

for i in range(0, 100):
    testLabels.append([0, 0, 0, 1, 0, 0, 0, 0, 0])

for i in range(0, 100):
    testLabels.append([0, 0, 0, 0, 1, 0, 0, 0, 0])

for i in range(0, 100):
    testLabels.append([0, 0, 0, 0, 0, 1, 0, 0, 0])

for i in range(0, 100):
    testLabels.append([0, 0, 0, 0, 0, 0, 1, 0, 0])

for i in range(0, 100):
    testLabels.append([0, 0, 0, 0, 0, 0, 0, 1, 0])

for i in range(0, 100):
    testLabels.append([0, 0, 0, 0, 0, 0, 0, 0, 1])

# Define the CNN Model
tf.compat.v1.reset_default_graph()
convnet=input_data(shape=[None,89,100,1],name='input')
convnet=conv_2d(convnet,32,2,activation='relu')
convnet=max_pool_2d(convnet,2)
convnet=conv_2d(convnet,64,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,128,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,256,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,256,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,128,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,64,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=fully_connected(convnet,1000,activation='relu')
convnet=dropout(convnet,0.75)


#TODO: Change the 2nd parameter to number of classes
convnet=fully_connected(convnet,9,activation='softmax')

convnet=regression(convnet,optimizer='adam',learning_rate=0.001,loss='categorical_crossentropy',name='regression')

model=tflearn.DNN(convnet,tensorboard_verbose=0)

# Shuffle Training Data
loadedImages, outputVectors = shuffle(loadedImages, outputVectors, random_state=0)

# Train model
#TODO: Change Epoch Back to 50
model.fit(loadedImages, outputVectors, n_epoch=50,
           validation_set = (testImages, testLabels),
           snapshot_step=100, show_metric=True, run_id='convnet_coursera')

model.save("TrainedModel/GestureRecogModel.tfl")