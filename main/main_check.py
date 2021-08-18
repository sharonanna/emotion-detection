import cv2
import numpy as np
import os
left_eye_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default')
for dirname, dirnames, filenames in os.walk("train/"):
    print dirname, dirnames, filenames
    for subdirname in filenames:
        print subdirname[0]
        path_name = os.path.join(dirname, subdirname)
        print path_name
        rgb_img = cv2.imread(path_name)
        gray_image = cv2.cvtColor(rgb_img,cv2.COLOR_BGR2GRAY)
        gray_image1 = gray_image
        feat_last = []
        flag1 = 0
        flag2 = 0
        flag3 = 0
        flag4 = 0


        eye_rects = left_eye_cascade.detectMultiScale(gray_image)
        for (x,y,w,h) in eye_rects:
            if flag1 == 0:
                print x,y,w,h
                cv2.rectangle(gray_image1,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray_image[y:y+h, x:x+w]
                cv2.imshow('eye',roi_gray)
                flag1 = 1

        eye_rects = right_eye_cascade.detectMultiScale(gray_image)
        for (x,y,w,h) in eye_rects:
            if flag4 == 0:
                print x,y,w,h
                cv2.rectangle(gray_image1,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray_image[y:y+h, x:x+w]
                cv2.imshow('eye',roi_gray)
                flag4 =1


        mouth_rects = mouth_cascade.detectMultiScale(gray_image, 1.3, 11)
        for (x,y,w,h) in mouth_rects:
            if flag2 == 0:
                cv2.rectangle(gray_image1,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray_image[y:y+h, x:x+w]
                cv2.imshow('mouth',roi_gray)
                flag2 = 1

        nose_rects = nose_cascade.detectMultiScale(gray_image)
        print nose_rects
        for (x,y,w,h) in nose_rects:
            if flag3 == 0:
                cv2.rectangle(gray_image1,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray_image[y:y+h, x:x+w]
                cv2.imshow('nose',roi_gray)
                flag3 = 1
        print flag1,flag2,flag3,flag4
        cv2.imshow('figure',gray_image)
        print feat_last
        cv2.waitKey(1000)
