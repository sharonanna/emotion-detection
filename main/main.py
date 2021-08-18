__author__ = 'Unnikrishnan'
import cv2
import numpy as np

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
    print im

    div = 40
    feature=np.array([])
    for s in range(0,height,height):
        for t in range(0,width,width):
            lbp_feat = im[s:s+div,t:t+div]
            feat = np.histogram(lbp_feat.ravel(),uni)
            feat1 = feat[0]
            # print feat1
            feature = np.concatenate((feature,feat1))
    return feature


rgb_img = cv2.imread('KA.AN1.39.tiff')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
gray_image = cv2.cvtColor(rgb_img,cv2.COLOR_BGR2GRAY)
gray_image1 = gray_image
eye_rects = eye_cascade.detectMultiScale(gray_image)
feat_last = []
for (x,y,w,h) in eye_rects:
    print x,y,w,h
    cv2.rectangle(gray_image1,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray_image[y:y+h, x:x+w]
    cv2.imshow('eye',roi_gray)
    feat = lbp_feat(roi_gray)
    feat_last.append(feat)

mouth_rects = mouth_cascade.detectMultiScale(gray_image, 1.3, 11)
for (x,y,w,h) in mouth_rects:
    cv2.rectangle(gray_image1,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray_image[y:y+h, x:x+w]
    cv2.imshow('mouth',roi_gray)
    feat = lbp_feat(roi_gray)
    feat_last.append(feat)

nose_rects = nose_cascade.detectMultiScale(gray_image)
print nose_rects
for (x,y,w,h) in nose_rects:
    cv2.rectangle(gray_image1,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray_image[y:y+h, x:x+w]
    cv2.imshow('nose',roi_gray)
    feat = lbp_feat(roi_gray)
    feat_last.append(feat)
cv2.imshow('figure',gray_image)
print feat_last
cv2.waitKey(0)