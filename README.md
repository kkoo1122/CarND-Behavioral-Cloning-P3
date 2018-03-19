# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains project files for the Behavioral Cloning Project.

This project used deep neural networks and convolutional neural networks to clone driving behavior. The model will output a steering angle to an autonomous vehicle.  The neural network architecture is based on nVidia paper (https://arxiv.org/abs/1604.07316) which train a CNN to map raw pixels from a single front-facing camera directly to steering commands.  It has proven to be successful in car self-driving enviroment so it should be proper in this project. 

We use Udacity car simulator to collect view image data of the center, left and right camera, while it also records the steering angle, throttle, and speed of the driving car at the same time.   We also did data balance and image augment to increase training data quality and quantity.  After trained, the model can drive the car automatically to run on the road.


## Project Goals
* Use the simulator to collect data of good driving behavior
* Preprocess and augment collected data
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## File Included
* model.py: containing the script to create and train the model
* drive.py: for driving the car in autonomous mode
* model.h5: containing a trained convolution neural network 
* README.md: summarizing the results
* enviroment.yml and enviroment-gpu.yml: enviroment build script
* drive1.mp4: the video of the first track run
* drive2.mp4: the video of the second track run

## Code Excution
### 1. enviroment
I trained model on Ubuntu 14/GTX 1080 and run car simulator on Mac OSX.  To make sure these two platforms have same software version and binary compatible, I used enviroment.yml and enviroment-gpu.yml to create my training and running environment by executing
```sh
$ conda env create -f enviroment.yml 
```
or 

```sh
$ conda env create -f enviroment-gpu.yml 
```

### 2. training the model
I used Keras to build deep neural networks.  To train the model, please place collected data under ./data/ directory and ./data/driving_log.csv should exist.  Then train the model by executing
```sh
$ python model.py
```

### 3. car self-driving automatically
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
$ python drive.py model.h5
```

## Camera View and Perspective Model

### 1. perspective view parameter assumption
Car simulator will record three camera veiws: center, left, and right views, in the training mode.  But only center view is given in autonomous driving mode.   We can extend our training sets by transforming left and right views to center view based on perspective projection calculation.  However, we don't exactly know the camera positions and projection matrix so we must do some assumptiona and use some trick to estimate these parameters and transformation formula. 

We used the bridge landscape shown in the following because this bridge edges are parallel straigt lines.  We assume the car width is average width 1.9M.  Comparing to car width, we can estimate the bridge width is about 7.6M.  We can assume the center camera is positioned at the center of the car. The left and right cameras should be on the side of the car.  Here, we asumme the left and right cameras are aparted from the center camera from 0.9M, roughlt the half width of car width.    The height of camera should be roughly the same as our eyes's height in driving. We assume the camera height is about 1.5M.  

<img src="./examples/car_bridge.png" width="480" alt="car bridge" />

Based on these parameters assumption, we can project the road view onto the camera screen shown in the following.   We let the y-position of the camera be zero for convenience.  So the road is on the plane at y = -1.5M.    We also let the x-position of the center camera be zero so the x-poistion of the left camera and right camera are -0.9M and 0.9M, respectively. 

<img src="./examples/car3dview2.png" width="480" alt="car 3d view" />

Car simulator records camera views shown in the following. The cross of two red lines are vanishing point of parallel lines.  They are all vanished at the same point, say (150, 60) in the captured images, in all three views.  The scene above it (blue line) is assumed at infinity or sky which we are not interested.  The scene below the blue line is assumed on the same road plane, say y = -1.5M.  The assumption may introduce some mapping distorion but it should not affect our training a lot. 

<table border="1">
<tr>
<td><img src="./examples/left_1.png" width="300" alt="left view" /></td>
<td><img src="./examples/center_1.png" width="300" alt="center view" /></td>
<td><img src="./examples/right_1.png" width="300" alt="right view" /></td>
</tr>
<tr>
<td><center>left view</center></td>
<td><center>center view</center></td>
<td><center>right view</center></td>
</tr>
</table>

### 2. transformation for camera shift

Based on the above assumption, we can estimate the x and y coordinates of the object and camera.  However, we still not know the z coordinate (z-depth) of them.   We apply the following tricks to estimate them. As shown in the following, we mark four landmarks.  These landmarks are easy to tracked in the projected center, left and right views.  Accoring to these corresponing points and perspective projection matrix, we can get the transform formula of the car horizontal shift. 

<table border="1">
<tr>
<td align="bottom"><img src="./examples/car_points.png" width="480" alt="car points" /></td>
<td align="bottom"><img src="./examples/center_2.png" width="300" alt="car points after projected"/></td>
</tr>
<tr>
<td><center>perspective view</center></td>
<td><center>center view</center></td>
</tr>
</table>

After math reduction, we get a quite simple transformation formula for mapping camera shift -0.9M, which map left camera view to center view.

<img src="./examples/eq1.png">

, where x, and y are the x- and y-coordinates in the captured view image, which range from (0, 0) to (300, 160). Parameter &alpha; is found the linear regresssion fitting the above landmarks.  The &alpha; is found as 0.72. 

The following gives the examples to apply these formula to map the left and right camera view to the center camera position.  After shifting the left and right camera view to the center position, the transformed images are quite matched to the center image. 

<table border="1">
<tr>
<td><img src="./examples/left_0.jpg" width="300" alt="original left view" /></td>
<td><img src="./examples/center_0.jpg" width="300" alt="original center view" /></td>
<td><img src="./examples/right_0.jpg" width="300" alt="original right view" /></td>
</tr>
<tr>
<td><center>left view</center></td>
<td><center>center view</center></td>
<td><center>right view</center></td>
</tr>
<tr>
<td><img src="./examples/left_2.jpg" width="300" alt="mapped left view to center camera" /></td>
<td><img src="./examples/center_0.jpg" width="300" alt="original center view" /></td>
<td><img src="./examples/right_2.jpg" width="300" alt="mapped right view to center camera" /></td>
</tr>
<tr>
<td><center>mapped left view to center camera</center></td>
<td><center>original center view</center></td>
<td><center>mapped right view to center camera</center></td>
</tr>
</table>

### 3. transformation for camera view rotation

The camera view will do horizontal translation while car or camera do a small angle translation.  It is useful for the following data augment.  We estimate the corresponding relationship between the translation scale and the rotation angle based the following captured image. The width 2m is got by comparing car width.  The depth 4m is got by the bridge width substracting the car trunk size.  We can get that translating 1 pixel is roughly to rotate 0.5 degress angle.   

<img src="./examples/rotation.png" width="300" alt="camera rotation" />


## Data Preprocessig

### 1. data balance

There are 8037 captured driving data.  Each drivng datum has center view, left view, right view, steering angle, throttle, and speed.   However, a lot of driving data are adjust driving forword with steering angle equal to zero.  It is not good for training because of data set bias.  To solve it, we calculate the histgram of the steering angle first; the we cut the number of the largest bins to the same number of the second largest bins.   The results show as follows:

<table border="1">
<tr>
<td><img src="./examples/bias_hist.png" width="400"/></td>
<td><img src="./examples/balance_hist.png" width="400"/></td>
</tr>
<tr>
<td><center>before data balance</center></td>
<td><center>after data balance</center></td>
</tr>
</table>

### 2. view image cropping and resizing

The size of the captured view image is 320 by 160.  The part above y=60 is infinity which we are not interested.  Even the part between y=60 and y=70 is quite far way distance which may not help a lot in training.  So we crop the image by the left-top point (0, 70) to right-bottom point (320, 136).  Then we resize it to 200 by 66 because the neural network input of nVidia architecture is 200 by 66 too.

The following shows the cropped results.

<table border="1">
<tr>
<td><img src="./examples/crop_region.jpg" width="480"/></td>
<td><img src="./examples/crop_resize.jpg" width="300"/></td>
</tr>
<tr>
<td><center>cropped region</center></td>
<td><center>after cropped and resized</center></td>
</tr>
</table>

## Data Augmentation

The following data augmentation are applied to increase training set and avoid overfitting.

## 1. brightness adjustment

We convert the RGB image into HSV and adjust V value randomly and convert back to RGB domain.  The following show the results.

<table border="1">
<tr>
<td><img src="./examples/center_0.jpg" width="300"/></td>
<td><img src="./examples/center_brightness.jpg" width="300"/></td>
</tr>
<tr>
<td><center>original</center></td>
<td><center>brightness augmentation</center></td>
</tr>
</table>

## 2. fake shadow

Add adjust part of brightness to create fake shadow for training.  The following show the results.

<table border="1">
<tr>
<td><img src="./examples/center_0.jpg" width="300"/></td>
<td><img src="./examples/center_shadow.jpg" width="300"/></td>
</tr>
<tr>
<td><center>original</center></td>
<td><center>shadow augmentation</center></td>
</tr>
</table>

## 2. mirror augment

Just mirror the captured image and inveres the steering angle.  The following show the results.

<table border="1">
<tr>
<td><img src="./examples/center_0.jpg" width="300"/></td>
<td><img src="./examples/center_mirror.jpg" width="300"/></td>
</tr>
<tr>
<td><center>original</center></td>
<td><center>mirror augmentation</center></td>
</tr>
</table>

## 3. left and right camera view augment

Car driving simulator records center, left and right images at the same time.  We can applied left and right captured image to train model.   It is intuitive to steer right if the car is too left and steer left if the car is too right. But we don't have proper steering angle for then.  Proper steering angle may depends on the car speed and how quick to make the car to drive back to the center line.   

To estimate the steering angle adjustment for the left and right view, we adopt the camera shift and camera rotation formula.   We knew the camera shift 0.9M in left and right view.  Then we apply camera rotation to let their pixel shifts are rought eqaul in the middle region of the scene.  Then we estimate the steering angle is about 5 degree adjustment for the left and right side view. This value is not far away from our driving experience.  In our steering angle parameter, the maximun is 1, which is corresponding to 25 degree.  So we got steering adjustmnet value 0.2 for the left and right view image.   They are shown in the following.

<table border="1">
<tr>
<td><img src="./examples/center_0.jpg" width="300" alt="original center view" /></td>
<td><img src="./examples/left_0.jpg" width="300" alt="original left view" /></td>
<td><img src="./examples/right_0.jpg" width="300" alt="original right view" /></td>
</tr>
<tr>
<td><center>center view</center></td>
<td><center>left view, steering angle is adjusted by 0.2</center></td>
<td><center>right view, steering angle is adjusted by -0.2</center></td>
</tr>
</table>

## 4. camera shift augmentation

Besides using the left and right camera image, we can apply camera shift formula to create augmentation image.  After apply camera shift transformation, we adjust steering angle correspondingly.  The following show an example.

<table border="1">
<tr>
<td><img src="./examples/center_0.jpg" width="300" alt="original center view" /></td>
<td><img src="./examples/center_shift.jpg" width="300"/></td>
</tr>
<tr>
<td><center>center view</center></td>
<td><center>camera shift left 1 meter, and <br>steering angle is adjusted by 0.22</center></td>
</tr>
</table>

## 5. camera rotation augmentation

Based on the camera transformation discussed in the previous section, we can generate camera rotation augmentation by translating the capture image horizontally.   After apply camera shift transformation, we adjust steering angle correspondingly.  In order not to over-turning, we adjust steering with half of the turning angle.  The following show an example.

<table border="1">
<tr>
<td><img src="./examples/center_0.jpg" width="300" alt="original center view" /></td>
<td><img src="./examples/center_rotate.jpg" width="300"/></td>
</tr>
<tr>
<td><center>center view</center></td>
<td><center>camera turns left by 10 degree, and <br>steering angle is adjusted by 0.2</center></td>
</tr>
</table>


### Model Architecture and Training Strategy

#### 1. The nVidia model architecture has been employed

The nVidia model architecture is adopted.  The following shows the network, including a input layer, 5 convolutional layers, and 3 fully connected layers.  The RELU activation function is used in all layers except the last layer use sigmoid.  No dropout layers are used.   We used Tensorflow and Keras framework to implement this model.

<img src="./examples/nvidia.png" width="480" alt="nvidia architecture" />
(Sources: [Nvidia, MitTechReview, Nvidia Blog](https://devblogs.nvidia.com/deep-learning-self-driving-cars/))

#### 2. Methods to reduce overfitting in the model

The model didn't use dropout layers in order to reduce overfitting. Instead, we used a lot of data augmentation to avoid overfitting.

#### 3. Model parameter tuning

The model used an adam optimizer and mse loss function.  The learning rate was not tuned manually.

#### 4. Appropriate training data

There are about eight thousands pieces of driving data, including center image, left image, right image, steering, throttle, brake, and speed values.   

In this project, our model only trained steering value accoring the input image.  Throttle value is automatically adjust by a PID controller.  Brake and speed value is not used in the training processs.  

A lot of data is driving forward with steering = 0, which will cause training bias.  We did 
data balance by cutting the number of them.  After data balance, there are about four thousands data set. The 25% of the data are split as our validation data.  Only the center image is used for our validation.  The left and right can be used as data augmention but not proper for data validation. 

The rest 75% of the data are used for training.   Besides the center image, the left and right image are used in training as discussed in the previous data augmentation section. Data augmentation methods, such as brightness, shadow, mirror, cam shift, and cam rotation, are applied to the center, left, right images randomly.  We used Keras fit_generator function to generate these augmetation data dynamically.

#### 5. Throttle and brake strategy

We didn't train the throttle value. Instead, we used a PID controller to controll throttle value given a speed.  The speed may varies during running.  When the car is running straight, say steering angle is 0 or small, we would like run fast.  On the other side, when the can is turning,say steering angle is large, we would like run slow.  We dynamic adjust speed from the maximum speed 12 MPH to the minimum speed 6 MPH.

The car turns on brake when throttle is negative.   However, we would not like to brake the car too often.  We did a littfle adjustment, when the throttle is a little below 0, say betwteen 0 and -0.4, just let throttle be 0.   When the throttle is below -0.4, the brake turn on by letting throttle be (throttle+0.4)

### Training processing and results

### 1. training for track1

It is not difficult to make car run full rounds of the first track after a few epoches training.  The following shows the result video.  Click it for viewing the full video.

[![Watch the video](https://raw.github.com/GabLeRoux/WebMole/master/ressources/WebMole_Youtube_Video.png)](http://youtu.be/vt5fpE0bzSY)  

