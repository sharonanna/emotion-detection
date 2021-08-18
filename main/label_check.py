import os
# AN=1, DI=2, FE=3, HA=4, NE=5,SU=6, SA=7
label = []
for dirname, dirnames, filenames in os.walk("train/"):
    print dirname, dirnames, filenames
    feat_last = []
    for subdirname in filenames:
        # print subdirname[0]
        path_name = os.path.join(dirname, subdirname)
        label1 = subdirname.split('.')
        if label1[1][0:2]=='AN':
            label.append(1)
        if label1[1][0:2]=='DI':
            label.append(2)
        if label1[1][0:2]=='FE':
            label.append(3)
        if label1[1][0:2]=='HA':
            label.append(4)
        if label1[1][0:2]=='NE':
            label.append(5)
        if label1[1][0:2]=='SU':
            label.append(6)
        if label1[1][0:2]=='SA':
            label.append(6)
    print label
