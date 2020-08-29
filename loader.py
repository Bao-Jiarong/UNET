'''
  Author       : Bao Jiarong
  Creation Date: 2020-08-28
  email        : bao.salirong@gmail.com
  Task         : load data
'''
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import random
import cv2

def scaling_tech(x,method = "normalization"):
    if method == "normalization":
        x = (x-x.min())/(x.max()-x.min()+0.0001)
    else:
        x = (x-np.std(x))/x.mean()
    return x

def load_heavy(dir,width,height,shuffle=True, split_ratio = 0.8, augment_data = False):
    # Step 1: Load Data...............
    # subdirs = os.listdir(dir)
    # subdirs = [d for d in subdirs if os.path.isdir(dir+d)]
    # subdirs = sorted(subdirs)

    labels = []
    images = []
    img_ext = [".jpg",".png",".jpeg"]

    print("Loading data")
    # n = len(subdirs)
    # for i in range(n):
    #     filenames = os.listdir(dir+subdirs[i])
    #     print("class",subdirs[i],"contains",len(filenames),"images")
    filenames = os.listdir(dir)
    L = len(filenames)
    print("contains",L,"images")
    for i in range(L):
        print(i+1,"/",L,end="           \r"); sys.stdout.flush()
        ext = (os.path.splitext(filenames[i]))[1]
        if ext in img_ext:
            # img_filename = dir + subdirs[i]+"/" + filenames[j]
            img_filename = os.path.join(dir, filenames[i])
            img   = cv2.imread(img_filename)
            w = img.shape[1] >> 1
            # print(img.shape)
            colored = img[:,:w,:]
            colored = cv2.resize(colored,(height,width),interpolation = cv2.INTER_AREA)/255.0
            sketch  = img[:,w:,:]
            sketch = cv2.resize(sketch,(height,width),interpolation = cv2.INTER_AREA)/255.0

            images.append(colored.astype(np.float32))
            labels.append(sketch.astype(np.float32))

            # data_augmentation
            if augment_data == True:
                colored_v = colored[:,::-1]
                sketch_v = sketch[:,::-1]
                # img_h = image[::-1,:]
                # img_h_v = image[::-1,::-1]
                x.append(colored_v)
                y.append(sketch_v)

    # images = np.array(images)
    # labels = np.array(labels)

    # Step 2: Normalize Data..........
    # images = scaling_tech(images,method="normalization")

    # Step 4: Shuffle Data............
    if shuffle == True:
        print("Shuffling data")
        indics = np.arange(0,len(images))
        np.random.shuffle(indics)

        labels = labels[indics]
        images = images[indics]

    # Step 5: Split Data.............
    print("Splitting data")
    m = int(len(images)*split_ratio)

    return images[:m], labels[:m], images[m:], labels[m:]

def load_light(dir,width,height,shuffle=True, split_ratio = 0.8, augment_data = False):
    # Step 1: Load Data...............
    # subdirs = os.listdir(dir)
    # subdirs = [d for d in subdirs if os.path.isdir(dir+d)]
    # subdirs = sorted(subdirs)

    labels = []
    images = []
    img_ext = [".jpg",".png",".jpeg"]

    print("Loading data")
    # n = len(subdirs)
    # for i in range(n):
    #     filenames = os.listdir(dir+subdirs[i])
    #     print("class",subdirs[i],"contains",len(filenames),"images")
    filenames = os.listdir(dir)
    print("contains",len(filenames),"images")
    for i in range(len(filenames)):
        ext = (os.path.splitext(filenames[i]))[1]
        if ext in img_ext:
            # img_filename = dir + subdirs[i]+"/" + filenames[j]
            img_filename = os.path.join(dir, filenames[i])

            images.append(img_filename)
            labels.append(i)

    images = np.array(images)
    labels = np.array(labels)

    # Step 2: Normalize Data..........
    # images = scaling_tech(images,method="normalization")

    # Step 3: Shuffle Data............
    if shuffle == True:
        print("Shuffling data")
        indics = np.arange(0,len(images))
        np.random.shuffle(indics)

        labels = labels[indics]
        images = images[indics]

    # Step 4: Split Data.............
    print("Splitting data")
    m = int(len(images)*split_ratio)

    return images[:m], labels[:m], images[m:], labels[m:]

def randfloat(low,high):
    return low+((high-low)*random.random())

def get_batch_heavy(X_train, Y_train, batch_size):
    n = len(X_train)
    t = int(randfloat(0,(n-batch_size+1)))
    x = np.asarray(X_train[t:t+batch_size])
    y = np.asarray(Y_train[t:t+batch_size])
    return x,y

def get_batch_light(X_train, Y_train, batch_size, height,width, augment_data = False):
    x = []
    y = []
    n = len(X_train)
    t = int(randfloat(0,(n-batch_size+1)))

    for i in range(t,t+batch_size):
        img   = cv2.imread(X_train[i]).astype(np.float32)
        w = img.shape[1] >> 1
        # print(img.shape)
        real = img[:,:w,:]
        real = cv2.resize(real,(height,width),interpolation = cv2.INTER_AREA)/255.0
        seg  = img[:,w:,:]
        seg = cv2.resize(seg,(height,width),interpolation = cv2.INTER_AREA)/255.0

        x.append(real)
        y.append(seg)

        if augment_data == True:
            img_v = real[:,::-1]
            seg_v = seg[:,::-1]
            # img_h = image[::-1,:]
            # img_h_v = image[::-1,::-1]
            x.append(img_v)
            y.append(seg_v)

    if augment_data == True:
        indics = np.arange(0,2*batch_size)
        np.random.shuffle(indics)
        x = x[indics]
        y = y[indics]
        x = x[:batch_size]
        y = y[:batch_size]

    return np.asarray(x), np.asarray(y)
