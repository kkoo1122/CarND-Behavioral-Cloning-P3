import os
import cv2
import random
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Conv2D, Cropping2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def loadData(basePath):
    ''' load data from csv
    '''
    fname = os.path.join(basePath, 'driving_log.csv')
    samples = pd.read_csv(fname)
    return samples


def balanceData(samples):
    """ crop the top part of the steering angle histogram

        use histgram and find the max bins and cut to the number of second max one
    """

    # calc histgram
    count, divs = np.histogram(samples.steering, bins=100)

    # find the max and the secon max bins
    idx = count.argsort()[::-1][:2]
    maxIdx = idx[0]
    maxN = count[idx[0]]
    cutN = count[idx[1]]
    
    # get the subset with maximum histgram
    maxSamples = samples[(samples.steering >= divs[maxIdx]) & (samples.steering < divs[maxIdx+1])]

    # random select rows to drop
    dropIdx = random.sample(list(maxSamples.index), maxN - cutN)
    balanceSamples = samples.drop(dropIdx)
    return balanceSamples


def brightness_change(image):
    """  change the brightness of the input image

    :param image: input image
    :return: new image
    """
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = np.random.uniform(0.2,0.8)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)

    return image1


def data_augmentation(images, angles):
    """ flip every image and change the blitheness of the image, then appended to the lists

    :param images: origin image
    :param angles: origin angles
    :return: added augmented images and their angles
    """
    augmented_images = []
    augmented_angles = []
    for image, angle in zip(images, angles):

        # print("imagex:", image, angle)
        augmented_images.append(image)
        augmented_angles.append(angle)

        # flip
        flipped_image = cv2.flip(image,1)
        flipped_angle = -1.0 * angle
        augmented_images.append(flipped_image)
        augmented_angles.append(flipped_angle)

        # brightness changes
        image_b1 = brightness_change(image)
        image_b2 = brightness_change(flipped_image)

        # append images
        augmented_images.append(image_b1)
        augmented_angles.append(angle)
        augmented_images.append(image_b2)
        augmented_angles.append(flipped_angle)

    return augmented_images, augmented_angles


def network_model():
    """

    :return: designed network model
    """
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(66,200,3)))
    model.add(Conv2D(24, (5,5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, (5,5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5,5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

def read_image(image_path):
    img = cv2.imread("data/"+image_path)
    img = img[...,::-1]     # change bgr to rgb
    img = img[70:70+66,:,:]
    img = cv2.resize(img, (200, 66))
    return img

def generator(samples, train_flag, batch_size=32):
    """

    """
    num_samples = len(samples)
    correction = 0.2  # correction angle used for the left and right images
    print("num_samples:", num_samples)

    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            for line in batch_samples:
                angle = float(line[3])
                c_imagePath = line[0].replace(" ", "")
                c_image = read_image(c_imagePath)
                images.append(c_image)
                angles.append(angle)

                '''
                cv2.imwrite("tmp.jpg", c_image)
                print("write image of", c_imagePath)
                exit()
                '''

                if train_flag:  # only add left and right images for training data (not for validation)
                    l_imagePath = line[1].replace(" ", "")
                    r_imagePath = line[2].replace(" ", "")
                    l_image = read_image(l_imagePath)
                    r_image = read_image(r_imagePath)
                    
                    images.append(l_image)
                    angles.append(angle + correction)
                    images.append(r_image)
                    angles.append(angle - correction)

            # flip image and change the brightness, for each input image, returns other 3 augmented images
            augmented_images, augmented_angles = data_augmentation(images, angles)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield shuffle(X_train, y_train)



# load the csv file
basePath = './data/'
print('loading the data...')
samples = loadData(basePath)

# balance the data
print('balance the data ...')
samples = balanceData(samples)

# split data into training and validation
train_samples, validation_samples = train_test_split(samples, test_size=0.3)

# compile and train the model using the generator function
train_generator = generator(train_samples, train_flag=True, batch_size=32)
validation_generator = generator(validation_samples, train_flag=False, batch_size=32)

# define the network model
MODEL_FILE = 'model.h5'
if path.isfile(MODEL_FILE):
    print("load model form", MODEL_FILE)
    model = load_model(MODEL_FILE)
else:
    print("create new model")
    model = network_model()
model.summary()

nbEpoch = 4
model.compile(loss='mse', optimizer='adam')

history = model.fit_generator(train_generator, steps_per_epoch=len(train_samples)//32, epochs=nbEpoch, 
    validation_data=validation_generator, validation_steps=len(validation_samples)//32)

model.save(MODEL_FILE)


