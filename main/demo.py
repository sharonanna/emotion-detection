import cv2
import numpy as np
import os
import pickle
# AN=1, DI=2, FE=3, HA=4, NE=5,SU=6, SA=7
left_eye_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

classifier = cv2.SVM()
classifier.load('svm_class.xml')

def dec2bin(n,num_bit):
    bin = []
    while(n!=1):
        bin.append(n%2)
        n = n/2
    bin.append(n%2)
    s = list(np.uint8(np.concatenate((bin,np.zeros(num_bit-len(bin)),[bin[0]]))))
    # s.reverse()
    return s

def table_loc():
    table = []
    for n in range(1,255):
        bin_val = np.array(dec2bin(n,8),dtype=float)
        # print type(bin_val)
        if (np.count_nonzero(np.diff(bin_val)))==2:
            table.append(n)
        else:
            table.append(5)
    table = np.array(table)
    loc = (table==5)
    table = np.concatenate(([0],table,[255,256]))
    uni = (np.unique(table))
    return loc,uni


def lbp_feat(img_last):
    # img_gray = cv2.resize(img_last,(160,160))
    img_gray = img_last
    cv2.waitKey(3000)
    lo,uni = table_loc()
    height, width = img_gray.shape
    init_img = np.uint8(np.zeros(shape=(height+2,width+2)))
    init_img[1:height+1,1:width+1]=img_gray
    lbp_img = np.zeros((height,width),dtype=np.float64)

    for i in range(1,height+1):
        for j in range(1,width+1):
            wind = init_img[i-1:i+2,j-1:j+2]
            thresh = init_img[i,j]
            blk2 = np.zeros((3,3))
            for k in range(0,3):
                for l in range(0,3):
                    if wind[k,l]>=thresh:
                        blk2[k,l] = 1
                    else:
                        blk2[k,l] = 0
            lb_bin = (np.concatenate(([blk2[0,0]],[blk2[0,1]],[blk2[0,2]],[blk2[1,2]],[blk2[2,2]],[blk2[2,1]],[blk2[2,0]],[blk2[1,0]])))
            # print lb_bin
            s = 0
            for ii in range(8):
                s = s + lb_bin[7-ii]*(2**(ii))
            # print s.dtype
            lbp_img[i-1,j-1] = s
    # print lbp_img

    im = lbp_img
    for nn in range(0,254):
        # print lo[nn]
        if lo[nn]:
            im[im==nn+1]=5
    # print im

    div = 40
    feature=[]
    for s in range(0,height,height):
        for t in range(0,width,width):
            lbp_feat = im[s:s+div,t:t+div]
            feat = np.histogram(lbp_feat.ravel(),uni)
            feat1 = feat[0]
            # print feat1
            feature.append(list(feat1))
            ggg = feature[0]
    return feature[0]

acc = []
for dirname, dirnames, filenames in os.walk("train/"):
    # print dirname, dirnames, filenames

    for subdirname in filenames:
        # print subdirname[0]
        path_name = os.path.join(dirname, subdirname)
        print path_name
        label1 = subdirname.split('.')
        if label1[1][0:2]=='AN':
            label = 1
        if label1[1][0:2]=='DI':
            label = 2
        if label1[1][0:2]=='FE':
            label = 3
        if label1[1][0:2]=='HA':
            label = 4
        if label1[1][0:2]=='NE':
            label = 5
        if label1[1][0:2]=='SU':
            llabel = 6
        if label1[1][0:2]=='SA':
            label = 7
        # print path_name
        rgb_img = cv2.imread(path_name)
        gray_image = cv2.cvtColor(rgb_img,cv2.COLOR_BGR2GRAY)
        gray_image1 = gray_image
        cv2.imshow('figure',gray_image)
        cv2.waitKey(1000)
        flag1 = 0
        flag2 = 0
        flag3 = 0
        flag4 = 0
        faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)
        # print faces
        feat_img = []
        for (x,y,w,h) in faces:
            roi_color = gray_image[y:y+h, x:x+w]
            eyes1 = right_eye_cascade.detectMultiScale(roi_color)
            for (ex,ey,ew,eh) in eyes1:
                # print ex,ey,ew
                if ex+ew/2<w/2 and ey+eh/2<h/2:
                    # cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                    # cv2.imshow('img',roi_color)
                    exl = int(ex)
                    eyl = int(ey)
                    ewl = int(ew)
                    ehl = int(eh)
                    # print ewl,ehl
                    crop1 = roi_color[eyl:eyl+ehl/2,exl+ewl/2:exl+ewl]
                    # cv2.rectangle(roi_color,(exl+ewl/2,eyl),(exl+ewl,eyl+ehl/2),(255,0,0),2)
                    # cv2.imshow('img',roi_color)
                    left_eye = roi_color[eyl:eyl+ehl, exl:exl+ewl]
                    # cv2.imshow('left',left_eye)
                    cv2.imshow('left1',crop1)
                    flag3 = 1

            eyes2 = left_eye_cascade.detectMultiScale(roi_color)
            for (ex,ey,ew,eh) in eyes2:
                # print ex,ey,ew
                if ex+ew/2>=w/2 and ey+eh/2<h/2:
                    exr = ex
                    eyr = ey
                    ewr = ew
                    ehr = eh
                    crop2 = roi_color[eyr:eyr+ehr/2,exr:exr+ewr/2]
                    # cv2.rectangle(roi_color,(exr,eyr),(exr+ewr/2,eyr+ehr/2),(255,0,0),2)
                    # cv2.imshow('img',roi_color)
                    # cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                    # cv2.imshow('img',roi_color)
                    right_eye = roi_color[eyr:eyr+ehr, exr:exr+ewr]
                    # cv2.imshow('right',right_eye)
                    cv2.imshow('right1',crop2)
                    flag4 = 1

            crop3 = roi_color[eyr+ehr/2:eyr+ehr,exl+ewl:exr]
            # cv2.rectangle(roi_color,(exl+ewl,eyr+ehr/2),(exr,eyr+ehr),(255,0,0),2)
            # cv2.imshow('img',roi_color)
            cv2.imshow('middle',crop3)

            mouth_rects = mouth_cascade.detectMultiScale(roi_color, 1.3, 11)
            for (ex,ey,ew,eh) in mouth_rects:
                if ey+eh/2>h-h/4:
                    if flag1 == 0:
                        exm = ex
                        eym = ey
                        ewm = ew
                        ehm = eh

                        # cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
                        # cv2.imshow('img',roi_color)
                        # mouth = roi_color[eym:eym+ehm, exm:exm+ewm]
                        # cv2.imshow('mouth',mouth)
                        flag1 = 1

            # cv2.rectangle(roi_color,(exm+ewm,eym),(exm+ewm+ewm/2,eym+ehm),(255,0,0),2)
            # cv2.imshow('img',roi_color)
            crop4 = roi_color[eym:eym+ehm,exm+ewm:exm+ewm+ewm/2]
            cv2.imshow('mouth1',crop4)
            # cv2.rectangle(roi_color,(exm-ewm/2,eym),(exm,eym+ehm),(255,0,0),2)
            # cv2.imshow('img',roi_color)
            crop5 = roi_color[eym:eym+ehm,exm-ewm/2:exm]
            cv2.imshow('mouth2',crop5)

            nose_rects = nose_cascade.detectMultiScale(roi_color, 1.3, 11)
            for (ex,ey,ew,eh) in nose_rects:
                if ey+eh/2>h/2:
                    if flag2 == 0:
                        # cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
                        # cv2.imshow('img',roi_color)
                        exn = ex
                        eyn = ey
                        ewn = ew
                        ehn = eh
                        # nose = roi_color[eyn:eyn+ehn, exn:exn+ewn]
                        # cv2.imshow('nose',nose)
                        flag2 = 1
            # cv2.rectangle(roi_color,(exn+ewn,eyn),(exn+ewn+ewn/2,eyn+ehn),(255,0,0),2)
            # cv2.imshow('img',roi_color)
            crop6 = roi_color[eyn:eyn+ehn,exn+ewn:exn+ewn+ewn/2]
            cv2.imshow('nose1',crop6)
            # cv2.rectangle(roi_color,(exn-ewn/2,eyn),(exn,eyn+ehn),(255,0,0),2)
            # cv2.imshow('img',roi_color)
            crop7 = roi_color[eyn:eyn+ehn,exn-ewn/2:exn]
            cv2.imshow('nose2',crop5)
            feat1 = list(lbp_feat(crop1))
            feat_img+=feat1
            feat2 = list(lbp_feat(crop2))
            feat_img+=feat2
            feat3 = list(lbp_feat(crop3))
            feat_img+=feat3
            feat4 = list(lbp_feat(crop4))
            feat_img+=feat4
            feat5 = list(lbp_feat(crop5))
            feat_img+=feat5
            feat6 = list(lbp_feat(crop6))
            feat_img+=feat6
            feat7 = list(lbp_feat(crop1))
            feat_img+=feat7
            feat_last = []
            feat_last.append(feat_img)
            f = np.array(feat_last, np.float32)
            t = classifier.predict(f)
            print t
            if t==1:
                print angry
                acc.append(1)
            else:
                acc.append(0)
            print acc


