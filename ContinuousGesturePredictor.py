import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import numpy as np
from PIL import Image
import cv2
import imutils


# global variables
bg = None

def resizeImage(imageName):
    basewidth = 100
    img = Image.open(imageName)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(imageName)

def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

def main():
    # initialize weight for running average
    aWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 225, 590

    # initialize num of frames
    num_frames = 0
    start_recording = False
    start_typing = False
    currentPrint = ""
    timer = 0;
    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width = 700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.Canny(gray ,50, 60)
        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                if start_recording or start_typing:
                    cv2.imwrite('Temp.png', thresholded)
                    resizeImage('Temp.png')
                    predictedClass, confidence = getPredictedClass()
                    if start_typing:
                        #Logic for telepromt here
                        currentPrint, timer = startTyping(predictedClass, confidence, currentPrint, timer)
                    else:
                        showStatistics(predictedClass, confidence)
                cv2.imshow("Thresholded", thresholded)

        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break

        #Start guessing the image
        if keypress == ord("s"):
            start_recording = True

        #Start Guessing image and typeing to telepromter
        if keypress == ord("t"):
            start_typing = True

        #Press c to clear
        if keypress == ord("c"):
            currentPrint = ""

        #Press space to space
        if keypress == 32:
            currentPrint += " "



def getPredictedClass():
    # Predict
    image = cv2.imread('Temp.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    prediction = model.predict([gray_image.reshape(89, 100, 1)])
    return np.argmax(prediction), (np.amax(prediction) / (prediction[0][0] + prediction[0][1] + prediction[0][2] + prediction[0][3] + prediction[0][4] \
                                                       + prediction[0][5] + prediction[0][6] + prediction[0][7] + prediction[0][8]))
    #return np.argmax(prediction), (np.amax(prediction) / (prediction[0][0] + prediction[0][1]  + prediction[0][2] + prediction[0][3]))

#Our own class for showing on the telepromt
def startTyping(predictedClass, confidence, currentPrint, timer):
    timer += 1
    textImage = np.zeros((512,512,3), np.uint8)
    addedLetter = ""

    if predictedClass == 0:
        addedLetter = "a"
    elif predictedClass == 1:
        addedLetter = "b"
    elif predictedClass == 2:
        addedLetter = "c"
    elif predictedClass == 3:
        addedLetter = "d"
    elif predictedClass == 4:
        addedLetter = "e"
    elif predictedClass == 5:
        addedLetter = "f"
    elif predictedClass == 6:
        addedLetter = "g"
    elif predictedClass == 7:
        addedLetter = "h"
    elif predictedClass == 8:
        addedLetter = "i"

    #If Confident about letter added to print
    if confidence >= .95 and timer > 60:
        currentPrint += addedLetter
        timer = 0

    cv2.putText(textImage, currentPrint,
    (20, 20),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (255, 255, 255),
    2)
    cv2.imshow("Text", textImage)
    return currentPrint, timer

def showStatistics(predictedClass, confidence):

    textImage = np.zeros((300,512,3), np.uint8)
    className = ""

    if predictedClass == 0:
        className = "Sign A"
    elif predictedClass == 1:
        className = "Sign B"
    elif predictedClass == 2:
        className = "Sign C"
    elif predictedClass == 3:
        className = "Sign D"
    elif predictedClass == 4:
        className = "Sign E"
    elif predictedClass == 5:
        className = "Sign F"
    elif predictedClass == 6:
        className = "Sign G"
    elif predictedClass == 7:
        className = "Sign H"
    elif predictedClass == 8:
        className = "Sign I"

    cv2.putText(textImage,"Predicted Class : " + className,
    (30, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2)

    cv2.putText(textImage,"Confidence : " + str(confidence * 100) + '%',
    (30, 100),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2)
    cv2.imshow("Statistics", textImage)




# Model defined
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

#TODO: change the second parameter to total number of classes
convnet=fully_connected(convnet,9,activation='softmax')

convnet=regression(convnet,optimizer='adam',learning_rate=0.001,loss='categorical_crossentropy',name='regression')

model=tflearn.DNN(convnet,tensorboard_verbose=0)

# Load Saved Model
model.load("TrainedModel/GestureRecogModel.tfl")

main()
