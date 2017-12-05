# **Behavioral Cloning** 

## By Jack Zhang
## For Udacity self-driving engineer nanodegree

### 11-30-2017

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./input_images.png "Sample of loaded images"
[image2]: ./input_cropped.png "Cropped images"
[image3]: ./input_hist.png "Histogram of input angles"
[image4]: ./center_all_hist.png "Histogram of original data"
[image5]: ./model.png "Final model"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

In addition I included the file load_data.py with my script for loading and visualizing the data

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the [NVIDIA end-to-end self-driving car neural network](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/).

It consists of a normalization layer (model.py lines 11-14) followed by 5 convolution layers, with 5x5/3x3 filter size, and depths between 24 and 64 (lines 15-23). Following the convolution layers are 3 fully connected layers, with sizes of 100, 50, and 10. 

The model includes ELU activation functions between each layer to introduce nonlinearity, while dropout layers (50% dropout rate) are implemented between each fully connected layer. The data is first cropped using a keras cropping layer (line 11) and then normalized using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 25, 28, 31, 34). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 42). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the training data provided by Udacity. In addition to the center camera images, I also used the left and right camera images, as well as horizontally flipped versions from all cameras. 

I also recorded several videos of my own on the training track, especially around the hard to navigate corners, for additional refinement data. In the final model, however, this augmented data was not used.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to utilize prior knowledge and experience with architectures that performed similar tasks. I started by trying transfer learning with pretrained models available through Keras, including ResNet, InceptionV3, VGG19, and MobileNet. I loaded these models without the top layers, and added my own dense layers underneath to train. I thought these models would provide good feature extraction for the driving scene, and the custom dense layers could use those features to determine the steering angles. However, this approach was very time consuming, as the models trained very slowly, and ultimately did not work well. I gave up trying transfer learning after several days and decided to build my own model in Keras.

For building my own model in Keras, I based the model on two different architectures, the one mentioned earlier by [NVIDIA](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/), and one by [comma.ai](https://github.com/commaai/research/blob/master/train_steering_model.py). I targeted these two architectures because they had been used previously and proven to work on similar tasks. 

I started by replicating the model by comma.ai, since it has less convolution layers than the NVIDIA model. In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. The model had a low error on the training set, but higher error on the validation set, which suggested overfitting. To combat this, I increased the dropout ratio in the model until the training and validation error was about the same. During testing, the car drove well but did not properly turn in 1 of 2 locations: the sharp left turn with the open right curb into the lake, and the sharp left turn after the bridge. I thought maybe the model didn't have enough depth to identify the features of this right curb, so I modified the model with an additional convolution layer, and also an additional dense layer. This helped the model achieve slightly lower training mean squared error, but the car still did not drive well around the two troublesome corners.

Finally, I decided to try out the NVIDIA model, which is similar to the comma.ai model, but with additonal convolution and fully connected layers. This model had slightly higher training and validation mean squared error than the comma.ai model, and it performed similarly to the previous model. I tried several variations of the NVIDIA model, including changing the input layer size, changing the convolution filter sizes, and increasing the dropout ratio. In the end, the biggest difference came when I changed the padding of the convolution from "SAME" to "VALID" padding. After many iterations of model refinement, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 10-36) consisted of a convolution neural network with the following layers and layer sizes:

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
--------------------------------------------------------------------
cropping2d_1 (Cropping2D)    (None, 80, 320, 3)        0         
_________________________________________________________________
lambda_1 (Lambda)            (None, 80, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 38, 158, 24)       1824      
_________________________________________________________________
elu_1 (ELU)                  (None, 38, 158, 24)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 17, 77, 36)        21636     
_________________________________________________________________
elu_2 (ELU)                  (None, 17, 77, 36)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 7, 37, 48)         43248     
_________________________________________________________________
elu_3 (ELU)                  (None, 7, 37, 48)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 5, 35, 64)         27712     
_________________________________________________________________
elu_4 (ELU)                  (None, 5, 35, 64)         0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 3, 33, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 6336)              0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 6336)              0         
_________________________________________________________________
elu_5 (ELU)                  (None, 6336)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               633700    
_________________________________________________________________
dropout_2 (Dropout)          (None, 100)               0         
_________________________________________________________________
elu_6 (ELU)                  (None, 100)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dropout_3 (Dropout)          (None, 50)                0         
_________________________________________________________________
elu_7 (ELU)                  (None, 50)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dropout_4 (Dropout)          (None, 10)                0         
_________________________________________________________________
elu_8 (ELU)                  (None, 10)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
--------------------------------------------------------------------
Total params: 770,619

Here is a visualization of the architecture

![Final model][image5]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I used the Udacity training data set. However, this data set is heavily skewed towards straight driving. Here is what the histogram of the steering angles look like.

![Center histogram][image4]

To reduce the bias towards zero angle steering, I randomly filtered out 90% of images that contained zero steering angle. For images that passed this filter, I also included images from the left and right cameras. These images had a correction factor of 0.1 added/subtracted to the steering angles. For all images that had non-zero steering angle, I included those in the training set, as well as their left and right camera counterpart. From all these images, I also included their flipped image in the horizontal direction. This produced ~20,000 images, which had a much nicer, more normal looking steering angle distribution. 

![Final histogram][image3]

Below, I show a sampling of the original training data set, which includes left and right cameras, as well as their flipped versions. The frame number and steering angle are included at the top of each image.

![Sample of original data set][image1]

In order to reduce the input size of the dataset as well as the remove features irrelevant for steering angle prediction, like the background, I cropped 60 pixels off the top and 20 pixels off the bottom of each image. The image below shows a random sample of cropped the cropped images. The cropping is applied in the model in a cropping layer (code line 11).

![Sample of cropped data set][image2]

After the collection process, I had ~20,000 different training images, the actual number depends on run to run because of the zero-angle filter. Since I was training on my desktop with a cpu, which had plenty of memory but did not have a capable gpu, I kept all the images in memory, in order to increase training time. 

I then preprocessed this data by dividing each image by 127.5 and subtracting 1. This was done in the model after the cropping layer, in a lambda layer (code line 12).

During training, I used the model.fit function of keras to split 20% of the data into a validation set. According to the keras documentation, this split is not random, so if I wanted to shuffle the data, I could have done so before model fitting. In the end, this was not used.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I set the number of training epochs to 10, and used an early stopping callback function to stop training if the validation loss increased over 2 training epochs. In the end, about 8 training epochs were used. I also used an adam optimizer so that manually training the learning rate wasn't necessary.
