# -*- coding: utf-8 -*-
"""

@author: abhilash
"""

import numpy as  np
import cv2

# load the image to detect, get width, height 
# convert to blob to pass into model
img_to_detect = cv2.imread('images/testing/scene5.jpg')
img_height = img_to_detect.shape[0]
img_width = img_to_detect.shape[1]

img_blob = cv2.dnn.blobFromImage(img_to_detect,swapRB=True,crop=False)
#convert BGR to RGB without cropping

# set of 90 class labels in predefined order
class_labels = ["person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light",
                "fire hydrant","street sign","stop sign","parking meter","bench","bird","cat","dog","horse",
                "sheep","cow","elephant","bear","zebra","giraffe","hat","backpack","umbrella","shoe","eye glasses",
                "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove",
                "skateboard","surfboard","tennis racket","bottle","plate","wine glass","cup","fork","knife",
                "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut",
                "cake","chair","sofa","pottedplant","bed","mirror","diningtable","window","desk","toilet","door","tv",
                "laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
                "blender","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]

# Loading pretrained model from buffer model weights and buffer config files
# input preprocessed blob into model and pass through the model
maskrcnn = cv2.dnn.readNetFromTensorflow('dataset/maskrcnn_buffermodel.pb','dataset/maskrcnn_bufferconfig.txt')
maskrcnn.setInput(img_blob)
(obj_detections_boxes,obj_detections_masks)  = maskrcnn.forward(["detection_out_final","detection_masks"])
# returned obj_detections_boxes[0, 0, index, 1] , 1 => will have the prediction class index
# 2 => will have confidence, 3 to 7 => will have the bounding box co-ordinates
no_of_detections = obj_detections_boxes.shape[2]

# loop over the detections
for index in np.arange(0, no_of_detections):
    prediction_confidence = obj_detections_boxes[0, 0, index, 2]
    # take only predictions with confidence more than 20%
    if prediction_confidence > 0.20:
        
        #get the predicted label
        predicted_class_index = int(obj_detections_boxes[0, 0, index, 1])
        predicted_class_label = class_labels[predicted_class_index]
        
        #obtain the bounding box co-oridnates for actual image from resized image size
        bounding_box = obj_detections_boxes[0, 0, index, 3:7] * np.array([img_width, img_height, img_width, img_height])
        (start_x_pt, start_y_pt, end_x_pt, end_y_pt) = bounding_box.astype("int")
        
        # print the prediction in console
        predicted_class_label = "{}: {:.2f}%".format(class_labels[predicted_class_index], prediction_confidence * 100)
        print("predicted object {}: {}".format(index+1, predicted_class_label))
        
        # draw rectangle and text in the image
        cv2.rectangle(img_to_detect, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), (0,255,0), 2)
        cv2.putText(img_to_detect, predicted_class_label, (start_x_pt, start_y_pt-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)


cv2.imshow("Detection Output", img_to_detect)










