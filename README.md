Project Author: Justin Nguyen, Vincent Mai, and Kaise Alnatour

Original Base Author: Sparsha Saha

# Hand Gesture Recognition Using Background Ellimination and Convolution Neural Network

## About the Project

This is a simple application of Convolution Neural Networks combined with background ellimination to detect different hand gestures. A background ellimination algorithm extracts the hand image from webcam and uses it to train as well predict the type of gesture that is. More information about the algorithm can be found below.

## Requirements

* Python 3.5-3.8
* pip 20.2.4
* TensorFlow 2.3.1
* tflearn 0.5
* sklearn 0.0
* imutils 0.5.3
* Pillow (PIL) 8.0.1
* OpenCV 4.4.0
* Numpy 1.18.5

## File Description

[PalmTracker.py]: Run this file to generate custom datasets. Go into the file and change the name of the directory and make other appropriate changes.

[ResizeImages.py]: Run this file after PalmTracker.py in order to resize the images so that it can be fed into the Convolution Neural Network designed using tensorflow. The network accepts 89 x 100 dimensional image.

[ModelTrainer.py]: This is the model trainer file. Run this file if you want to retrain the model using your custom dataset.

[ContinuousGesturePredictor.py]: Running this file opens up your webcam and takes continuous frames of your hand image and then predicts the class of your hand gesture in realtime.

## Some key architectural insights into the project

### Background Ellimination Algorithm

I have used opencv for taking a running average of the background for 30 frames and then use that running average to detect the hand that has to be introduced after the background has been properly recognized.

I had found a very useful article on foreground mask by [Gogul09](https://github.com/Gogul09) and i have pretty much used his code for background ellimination with a few changes in order to suit my cause.

He has written an awesome article on the problem and you can read it up [here](https://gogul09.github.io/software/hand-gesture-recognition-p1).

### The Deep Convolution Neural Network

The network contains **7** hidden convolution layers with **Relu** as the activation function and **1** Fully connected layer.

The network is trained across **50** iterations with a batch size of **64**.

I kind of saw that 50 iterations kind of trains the model well and there is no increase in validation accuracy along the lines so that should be enough.

The model achieves an accuracy of **96.6%** on the validation dataset.

The ratio of training set to validation set is **1000 : 100**.

## How to run the RealTime prediction

Run the [ContinuousGesturePredictor.py] file and you will see a window named **Video Feed** appear on screen. Wait for a while until a window named **Thresholded** appears.

The next step involves pressing **"s"** on your keyboard in order to start the real-time prediction.

Bring your hand in the **Green Box** drawn inside **Video Feed** window in order to see the predictions.
Look in demo for some visual clarity.

## How to run the RealTime ASL Image to text

Run the [ContinuousGesturePredictor.py] file and you will see a window named **Video Feed** appear on screen. Wait for a while until a window named **Thresholded** appears.

The next step involves pressing **"t"** on your keyboard in order to start the real-time prediction and translation.

A console box will appear showing you the current text you have created. Bring your hand in the **Green Box** drawn inside **Video Feed** window and gesture hand using the American Sign Lanaguge Alphabet using letters from (A-I). 

When a strong confidence occurs, it will convert that image into text in the console box. There is a buffer to allow time spacing between one gesture to the next. 

Pressing **"c"** will clear your current text in the console box. Pressing **" "** will add a space between characters in the text. 
