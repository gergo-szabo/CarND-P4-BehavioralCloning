# Self-Driving Car Engineer Nanodegree


## Project: **Behavioral Cloning** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around both track without leaving the road
(https://github.com/udacity/CarND-Behavioral-Cloning-P3)

Rubric: [link](https://review.udacity.com/#!/rubrics/432/view)

Helper files: [link](https://github.com/udacity/CarND-Behavioral-Cloning-P3)
Simulator: [link](https://github.com/udacity/self-driving-car-sim)

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* model.h5 containing a trained convolution neural network 
* run1.mp4 showing the agent on track one
* run2.mp4 showing the agent on track two
* writeup_report.md summarizing the results
* drive.py for driving the car in autonomous mode (see helper files, unchanged)
* video.py for creating video from images (see helper files, unchanged)

#### 2. Submission includes functional code
Using the drive.py file an the Udacity provided simulator's autonomus mode, the agent will drive the car around the tracks autonomously. Execute: 
```sh
python drive.py model.h5 run1
```

The third argument, model.h5, is the name of the saved NN model.

The fourth argument, run1, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network which is based on a network architecture published by Nvidia (https://devblogs.nvidia.com/deep-learning-self-driving-cars/).

Training data consists of 160x320 RGB pictures and steering angles.

1,  The data is normalized by Keras lamda layer.
2,  The top and bottom of the recorded images are cropped. (70/25 pixel)
3,  2D convolution: depth: 24, 5x5 kernel, 2x2 stride along pixels and RELU activation
4,  2D convolution: depth: 36, 5x5 kernel, 2x2 stride along pixels and RELU activation
5,  2D convolution: depth: 48, 5x5 kernel, 2x2 stride along pixels and RELU activation
6,  2D convolution: depth: 64, 3x3 kernel and RELU activation
7,  Droput: 0.5 (additon to original architecture)
8,  2D convolution: depth: 64, 3x3 kernel and RELU activation
9,  Droput: 0.5 (additon to original architecture)
10, Flatten (Dense)
11, Droput: 0.5 (additon to original architecture)
12, Dense: 100
13, Dense: 50
14, Dense: 1 (deviation to original architecture due to different output requirement)

#### 2. Attempts to reduce overfitting in the model

Introducing dropout seems to be unneccesery for the first few layer because the simulation has relatively low amount of variation in textures, features, etc.

The model contains dropout layers in it's middle section in order to reduce overfitting.

Last few layer has a relatively low amount of neuron so dropout was not implemented there.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The second part of track two is not present in the training and validation data. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

My training data consist of four recording of simulated driving:
1, 2 lap on first track. Center lane driving.
2, 1 run on first half of second track. Center lane driving.
3, 1 run on first half of second track. Driving close to inner curve of bends. Otherwise center lane driving.
4, 1 run on the first big curve of the second track. Driving in consciously unusual arcs. (Recovery action.)

![Begining of the 2nd half of track two][testset]

For details about how I processed the training data, see the next sections. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first goal was to create an agent which can drive around the first track. I decided to use the already mentioned Nvidia's architecture for autonomus driving with (unaugmented) training data from one lap of center lane driving. The result was very promising.

I collected more data from track two, started to use the left and right camera images and also mirrored every image (and steering angle).

The model performed well on the training data but not so much on the validation set. Also the car often went of road. Sometimes the NN learned to follow lane markings. This is not an acceptable solution for the first track where only the side lane markings are present. I think the problem was that the NN was overfitting for second track middle lane marking (2 epoch).

I implemented dropout layers for the middle layers. As I already mentioned my goal was to target the higher abstraction levels which have enough neuron per layer.

The model performed better than before but it was unable to drive through the first bigger curve of the second track. I decided to record additional data (Data 4: recovery action) and to teach the data more times (5 epoch).

The validation loss was not showing convergence to the training loss so I had concerns regarding the success of the training. However the agent performs perfectly on road segments which were not in the training data (second half of the second track)!

#### 2. Final Model Architecture

The final model architecture is already described in:
Model Architecture and Training Strategy: 1. An appropriate model architecture has been employed

#### 3. Creation of the Training Set & Training Process

The training data is randomly shuffled before training and 20% of the data is used as a validation set. As mentioned, the second half of track two was kept as a "test set".

I used an adam optimizer so that manually training the learning rate wasn't necessary.

Generator was not used because it was not neccesery. (i5-3320M 2x2600 MHz, 16 GB RAM)