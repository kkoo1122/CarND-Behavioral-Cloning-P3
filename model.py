import os
import cv2
import random
import numpy as np
import pandas as pd
from PIL import Image
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Conv2D, Cropping2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

DATAPATH = './data/'

def loadData():
    ''' load image and steering data from csv
    '''
    fname = DATAPATH + 'driving_log.csv'
    samples = pd.read_csv(fname)
    return samples

def balanceData(samples):
    """ crop the top part of the steering angle histogram

        use histgram and find the max bins and cut to the number of second max one
    """

    # calc histgram
    count, divs = np.histogram(samples.steering, bins=100)

    # find the max and the second max bins
    idx = count.argsort()[::-1][:2]
    maxIdx = idx[0]
    maxN = count[idx[0]]
    cutN = count[idx[1]]
    
    # get the subset with maximum histgram
    maxSamples = samples[(samples.steering >= divs[maxIdx]) & (samples.steering < divs[maxIdx+1])]

    # random select rows to drop so that the number of the maximum row is eqault to the second one
    dropIdx = random.sample(list(maxSamples.index), maxN - cutN)
    balanceSamples = samples.drop(dropIdx)
    return balanceSamples

def augmentBrightness(im):
    im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    random_bright = np.random.uniform(0.2,0.8)
    im[:,:,2] = im[:,:,2]*random_bright
    im = cv2.cvtColor(im, cv2.COLOR_HSV2RGB)
    return im

def augmentShadow(im):
    im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    ydim, xdim = im.shape[:2]
    x = np.linspace(0, 1, xdim)
    y = np.linspace(0, 1, ydim)
    xv, yv = np.meshgrid(x, y)
    shadow_value = np.random.uniform(0.2, 0.8)
    px1, py1 = np.random.rand(2, 1)
    px2, py2 = np.random.rand(2, 1)
    shadow = np.ones(xv.shape)
    shadow[(px2-px1)*(yv-py1)+(py2-py1)*(xv-px1)>0] = shadow_value
    im[:,:,2] = im[:,:,2] * shadow
    im = cv2.cvtColor(im, cv2.COLOR_HSV2RGB)
    return im

def augmentMirror(im):
    im = im[:,::-1,:]
    return im

def augmentShift(im, camShift):
    '''
        positive camShift is to shift right, steering angle adjustment is negative
    '''
    p = -camShift * 72
    rows, cols, ch = im.shape
    pts1 = np.float32([[0,0],[300,0],[0,100]])
    pts2 = np.float32([[0,0],[300,0],[p,100]])

    M = cv2.getAffineTransform(pts1,pts2)

    im2 = im.copy()
    im2[60:,:,:] = cv2.warpAffine(im[60:,:],M,(cols,rows-60))

    return im2

def augmentRotate(im, camDegree):
    
    '''
        positive camDegree is to turn right, steering angle adjustment is negative
    '''
    p = -int(camDegree * 2)
    im2 = np.zeros(im.shape, im.dtype)
    if p == 0:
        return im
    elif p > 0:
        im2[:,p:,:] = im[:,:-p,:]
    else:
        im2[:,:p,:] = im[:,-p:,:]

    return im2

def needAugment(augment):
    return np.random.uniform() > 0.5 if augment else False

def cropImage(im):
    im2 = im[70:136,:,:]
    im2 = cv2.resize(im2, (200, 66))
    return im2

def dataAugmentation(images, angles, augment):

    augmented_images = []
    augmented_angles = []
    for image, angle in zip(images, angles):

        # mirror
        if needAugment(augment):
            image = augmentMirror(image)
            angle = -angle

        # brightness & shadow
        if needAugment(augment):
            if needAugment(augment):
                image = augmentBrightness(image)
            else:
                image = augmentShadow(image)

        # shift & rotate
        if needAugment(augment):
            if needAugment(augment):
                # cam shift, max 1 meter
                cam_shift = np.random.uniform(-1, 1)
                image = augmentShift(image, cam_shift)
                angle -= cam_shift * 0.2 / 0.9
            else:
                # cam rotate, max 10 degree
                cam_degree = np.random.uniform(-10, 10)
                image = augmentRotate(image, cam_degree)
                angle -= 0.5 * cam_degree / 25

        # after augmented, angle should not over the range (-1, 1)
        if angle >= -1 and angle <= 1:
            augmented_images.append(cropImage(image))
            augmented_angles.append(angle)

    return augmented_images, augmented_angles


def create_model():
    '''
        to create nVidia car self-driving model
    '''
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(66,200,3)))
    model.add(Conv2D(24, (5,5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, (5,5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5,5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='tanh'))
    return model

def readImage(image_path):
    fname = DATAPATH + image_path
    im = np.array(Image.open(fname))
    return im

def generator(samples, data_augment, batch_size=32):

    num_samples = len(samples)

    # steering angle correction for left and right cameras
    correction = 0.2

    while 1:  # Loop forever so the generator never terminates
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            for _, row in batch_samples.iterrows():
                angle = row.steering
                c_image = readImage(row.center.strip())
                images.append(c_image)
                angles.append(angle)

                if data_augment:  
                    l_image = readImage(row.left.strip())
                    r_image = readImage(row.right.strip())
                    
                    images.append(l_image)
                    angles.append(angle + correction)
                    images.append(r_image)
                    angles.append(angle - correction)

            augmented_images, augmented_angles = dataAugmentation(images, angles, data_augment)
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
                
            yield shuffle(X_train, y_train)

# load the csv file
samples = loadData()
print('loading the data...')

# balance the data
samples = balanceData(samples)
print('balance the data ...')

# split data into training and validation
train_samples, validation_samples = train_test_split(samples, test_size=0.25)
print('split the data to train set={}, validate set={}'.format(
    len(train_samples), len(validation_samples)
))

# compile and train the model using the generator function
train_generator = generator(train_samples, data_augment=True, batch_size=128)
validation_generator = generator(validation_samples, data_augment=False, batch_size=32)

# define the network model
MODEL_FILE = 'model.h5'
if os.path.isfile(MODEL_FILE):
    print("load model form", MODEL_FILE)
    model = load_model(MODEL_FILE)
else:
    print("create new model")
    model = create_model()
model.summary()

model.compile(loss='mse', optimizer='adam')

nbEpoch = 10
for epoch in range(nbEpoch):
    print("epoch = {}/{}".format(epoch+1, nbEpoch))
    history = model.fit_generator(train_generator, steps_per_epoch=len(train_samples)//32, epochs=1, 
        validation_data=validation_generator, validation_steps=len(validation_samples)//32)
    
    # save model for echo epoch for debugging
    model.save('model_{}.h5'.format(epoch))

# save and override the trained model
model.save(MODEL_FILE)
print("model {} is saved".format(MODEL_FILE))

