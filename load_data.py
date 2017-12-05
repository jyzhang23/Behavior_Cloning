### Load data
# Udacity training data
# Remove 90% of zero steering images
# For all images being appended:
# Include left & right camera images
# Include horizontally flipped versions for all appended images

import csv
import cv2
import numpy as np
import os

path = os.getcwd()+'/data/'
lines = []
with open(path+'/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)

num_lines=len(lines)
lines=lines[1:num_lines]
images = []
steering = []

correction=0.1

for line in lines:
    current_steering = float(line[3])
    steering_left = current_steering + correction
    steering_right = current_steering - correction
    
    if current_steering == 0:
        if np.random.uniform()>0.90:
            image = cv2.imread(path + line[0])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #convert bgr to rgb
            images.append(image)
            steering.append(current_steering)
            
            # add left camera and flipped versions
            img_left = (cv2.imread(path + str.strip(line[1])))
            img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
            img_left_flip = cv2.flip(img_left,1)
            images.append(img_left)
            images.append(img_left_flip)
            steering.append(steering_left)
            steering.append(-steering_left) # reverse of flipped image

            # add right camera and flipped versions
            img_right = (cv2.imread(path + str.strip(line[2])))
            img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
            img_right_flip = cv2.flip(img_right,1)
            images.append(img_right)
            images.append(img_right_flip)
            steering.append(steering_right)
            steering.append(-steering_right) # reverse of flipped image
    else:
        image = cv2.imread(path + line[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #convert bgr to rgb

        images.append(image)
        steering.append(current_steering)
        
        # add left camera and flipped versions
        img_left = (cv2.imread(path + str.strip(line[1])))
        img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
        img_left_flip = cv2.flip(img_left,1)
        images.append(img_left)
        images.append(img_left_flip)
        steering.append(steering_left)
        steering.append(-steering_left) # reverse of flipped image
        
        # add right camera and flipped versions
        img_right = (cv2.imread(path + str.strip(line[2])))
        img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
        img_right_flip = cv2.flip(img_right,1)
        images.append(img_right)
        images.append(img_right_flip)
        steering.append(steering_right)
        steering.append(-steering_right) # reverse of flipped image
        
X_train=np.array(images)
y_train=np.array(steering)

# Check shape of arrays

print(X_train.shape)
print(y_train.shape)

### Data visualization
# Display 36 random images from training set

import matplotlib.pyplot as plt
from random import *

def plotSamples(data, labels):
    fig,axes=plt.subplots(6,6, figsize=(25,20),subplot_kw={'xticks':[],'yticks':[]})
    plt.suptitle('Random sample of training data')
    axes=axes.ravel()
    
    for i in range(0, 36):
        rand_index=randint(1,len(data))
        img=data[rand_index,...]
        axes[i].imshow(img)
        temp_str='#: '+str(rand_index)+' Angle: '+'{:6.5}'.format(str(labels[rand_index]))
        axes[i].set_title(temp_str)
    plt.show()
    #%matplotlib inline
    
plotSamples(X_train,y_train)

# Histogram of distribution

num_bins=50

plt.hist(y_train,num_bins,normed=0)

plt.show()
#%matplotlib inline
