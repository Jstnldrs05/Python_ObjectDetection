#Ladores, Justine
#BSIT 3-2 || INTECH 3201
# 1st and 2nd Term Project - Object Detection using OpenCV

import cv2 as cv
import numpy as np
from cv2 import waitKey

#img = cv.imread("grp2.jpg")

thres = 0.6 # threshold to detect objects
nms_threshold = 0.5

cap = cv.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)


classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')


"""
def rescaleFrame(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

img = rescaleFrame(img)
"""
#print(classNames)
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


while True:

    success,img = cap.read()

    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float,confs))
    #print(type(confs[0]))
    #print(confs)

    indices = cv.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)
    print(indices)
    
    for i in indices:
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv.rectangle(img, (x,y),(x+w,h+y), color=(0, 255, 0), thickness=2)
        cv.putText(img, classNames[classIds[0]-1].upper(), (box[0]+10,box[1]+30),
        cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)


    #if len(classIds) != 0:

     #   for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
      #      cv.rectangle(img, box, color=(0,255,0),thickness=2)
       #     cv.putText(img, classNames[classId-1].upper(), (box[0]+10,box[1]+30), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        #    cv.putText(img, str(round(confidence*100,2)), (box[0]+200,box[1]+30), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)




    cv.imshow("Output", img)
    cv.waitKey(1)