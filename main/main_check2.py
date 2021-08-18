import cv2
import numpy as np
import os
left_eye_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
for dirname, dirnames, filenames in os.walk("train/"):
    print dirname, dirnames, filenames
    for subdirname in filenames:
        # print subdirname[0]
        path_name = os.path.join(dirname, subdirname)
        # print path_name
        rgb_img = cv2.imread(path_name)
        gray_image = cv2.cvtColor(rgb_img,cv2.COLOR_BGR2GRAY)
        gray_image1 = gray_image
        cv2.imshow('figure',gray_image)
        cv2.waitKey(1000)
        feat_last = []
        flag1 = 0
        flag2 = 0
        flag3 = 0
        flag4 = 0
        faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)
        # print faces
        for (x,y,w,h) in faces:
            roi_color = gray_image[y:y+h, x:x+w]
            eyes1 = right_eye_cascade.detectMultiScale(roi_color)
            for (ex,ey,ew,eh) in eyes1:
                # print ex,ey,ew
                if ex+ew/2<w/2 and ey+eh/2<h/2:
                    # cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                    # cv2.imshow('img',roi_color)
                    left_eye = roi_color[ey:ey+eh, ex:ex+ew]
                    cv2.imshow('left',left_eye)
                    flag3 = 1

            eyes2 = left_eye_cascade.detectMultiScale(roi_color)
            for (ex,ey,ew,eh) in eyes2:
                # print ex,ey,ew
                if ex+ew/2>=w/2 and ey+eh/2<h/2:
                    # cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                    # cv2.imshow('img',roi_color)
                    right_eye = roi_color[ey:ey+eh, ex:ex+ew]
                    cv2.imshow('right',right_eye)
                    flag4 = 1


            mouth_rects = mouth_cascade.detectMultiScale(roi_color, 1.3, 11)
            for (ex,ey,ew,eh) in mouth_rects:
                if ey+eh/2>h-h/4:
                    if flag1 == 0:
                        # cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
                        # cv2.imshow('img',roi_color)
                        mouth = roi_color[ey:ey+eh, ex:ex+ew]
                        cv2.imshow('mouth',mouth)
                        flag1 = 1

            nose_rects = nose_cascade.detectMultiScale(roi_color, 1.3, 11)
            for (ex,ey,ew,eh) in nose_rects:
                if ey+eh/2>h/2:
                    if flag2 == 0:
                        # cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
                        # cv2.imshow('img',roi_color)
                        nose = roi_color[ey:ey+eh, ex:ex+ew]
                        cv2.imshow('nose',nose)
                        flag2 = 1
        # print flag1,flag2,flag3,flag4
        if flag1==0 or flag2 == 0 or flag3 == 0 or flag4 == 0:
            print path_name
        cv2.waitKey(0)
