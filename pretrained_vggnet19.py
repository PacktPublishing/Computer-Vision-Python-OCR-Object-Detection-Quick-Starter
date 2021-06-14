# -*- coding: utf-8 -*-
"""
@author: abhilash
"""
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import cv2

#loading the image to predict
img_path = 'images/test5.jpg'
img = load_img(img_path)

#resize the image to 224x224 square shape
img = img.resize((224,224))
#convert the image to array
img_array = img_to_array(img)

#convert the image into a 4 dimensional Tensor
#convert from (height, width, channels), (batchsize, height, width, channels)
img_array = np.expand_dims(img_array, axis=0)

#preprocess the input image array
img_array = imagenet_utils.preprocess_input(img_array)

#Load the model from internet / computer
#approximately 530 MB
pretrained_model = VGG19(weights="imagenet")

#predict using predict() method
prediction = pretrained_model.predict(img_array)

#decode the prediction
actual_prediction = imagenet_utils.decode_predictions(prediction)

print("predicted object is:")
print(actual_prediction[0][0][1])
print("with accuracy")
print(actual_prediction[0][0][2]*100)

#display image and the prediction text over it
disp_img = cv2.imread(img_path)
#display prediction text over the image
cv2.putText(disp_img, actual_prediction[0][0][1], (20,20), cv2.FONT_HERSHEY_TRIPLEX , 0.8, (0,0,0))
#show the image
cv2.imshow("Prediction",disp_img)



