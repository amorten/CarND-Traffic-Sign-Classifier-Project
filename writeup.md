# **Traffic Sign Recognition** 

## Writeup 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set 
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[explore_0]: ./writeup_images/explore_0.png "Visualization"
[explore_17]: ./writeup_images/explore_17.png "Visualization"
[color]: ./writeup_images/color.png "Before Grayscaling"
[gray]: ./writeup_images/gray.png "After Grayscaling"
[test_images]: ./writeup_images/test_images.png "Six More Traffic Signs"

[test0]: ./more_test_images/test_image_0.png  "Traffic Sign 1"
[test1]: ./more_test_images/test_image_1.png "Traffic Sign 2"
[test2]: ./more_test_images/test_image_2.png "Traffic Sign 3"
[test3]: ./more_test_images/test_image_3.png "Traffic Sign 4"
[test4]: ./more_test_images/test_image_4.png "Traffic Sign 5"
[test5]: ./more_test_images/test_image_5.png "Traffic Sign 6"

[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[notebook]: ./Traffic_Sign_Classifier.ipynb "IPython Notebook"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code][notebook] and the the [HTML version](./report.html).

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

I explore the data set in the [notebook][] in cell #3. For each of the 43 traffic sign classes, I print the class number, sign name, total number of samples of that class in the training data, and plot ten random sample images of that class.

For example, for class number 0, the sign name is "Speed limit (20km/h)", there are 180 samples, and here are ten random samples:

![sample data exploration][explore_0]

For class number 17, the sign name is "No entry", there are 990 samples, and here are ten random samples:

![sample data exploration][explore_17]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it would reduce the size of the input by a factor of three, which would speed up training.

Here is an example of a traffic sign image before and after grayscaling:

![color image][color]
![gray image][gray]

I then normalized the image data to be centered around zero and lie within the range [-1.0, 1.0]. Backpropagation of weights tends to work much better with normalized input.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is defined in cell #8 of the [notebook] by the function `LeNetBetterWithDropout` (the previous versions called `LeNet` and `LeNetBetter` defined in cells #6 and #7 were earlier less successful attempts). 

The model takes a normalized grayscale image as input. Then two convolutions are applied with ReLU and Max Pooling after each. The network is flattened, and the a Dropout layer is applied, with 50% of connections removed during training. That is followed by four fully-connected layers with ReLU activations after each layer except the final layer. The final output layer has a size of 43, the number of traffic sign classes.


`LeNetBetterWithDropout` consists of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input | 32x32x1 Grayscale image | 
| Convolution 5x5 | 1x1 stride, outputs 28x28x12 |
| RELU	|	|
| Max pooling	 | 2x2 stride,  outputs 14x14x12 |
| Convolution 5x5 | 1x1 stride, outputs 10x10x32 |
| RELU	| |
| Max pooling	 | 2x2 stride,  outputs 5x5x32 |
| Flatten | outputs	800 |
| Dropout | keep 50% during training
| Fully connected | outputs 400 |
| RELU	| |
| Fully connected | outputs 120 |
| RELU	| |
| Fully connected | outputs 84 |
| RELU	| |
| Fully connected | outputs 43 |



 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The model is trained by minimizing the softmax cross-entropy using and adam optimizer with learning rate 0.001. The batch size was 128, and the number of epochs was 10, because the validation accuracy did not increase significantly after about 10 epochs, and I wanted to avoid overfitting. I set up the model to have the dropout rate be a hyperparameter, which I set at 0.5.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:   

* training set accuracy of 0.969
* validation set accuracy of 0.956
* test set accuracy of 0.939
 
I started with the LeNet architecture because I had already implemented it in class to classify images and it is known to work well classifying small images with text-like patterns. Importantly, LeNet is a convolutional neural network. Convolutional layers should work well for this problem due to the translation invariance of the weights filter. The translation invariance is important for finding image features that could appear anywhere in the image.

In the initial LeNet architecture, there were fewer classes to predict in the output. I had to change it to 43. I also changed the number of color channels to three at first, because I was using color images, but later I changed it back to one channel after converting all images to grayscale.

Because the number of output classes was larger (43), I increased the depths of the convolutional layers. I was concerned about overfitting, so I also added a dropout layer after all of the convolutional layers were finished.

I played around with even deeper convolutional layers, and tried adding a third convolutional layer, but the effect in each case was to simply reduce the validation accuracy.

 I selected the number of epochs so that the training would stop when the validation accuracy began to appear constant. I tried a few values for the learning rate, finding that 0.001 gave the highest validation accuracy. I set the dropout probability at 50%, although setting it to 30% or 70% gave similar results.
  

 The model does well on the training, validation, and test data sets (all above 93% accuracy), but does not appear to generalize well to images outside of the data set. This suggests overfitting. I tried variable number of droupout layers in different locations and with different dropout probabilities, but nothing I tried helped the model with the additional test images found on the web.  I suspect the right way to solve this problem would be to instead modify the training images with rotation, skewing, and jittering to make the training more robust; or to add more completely indpendent images to the data set.
 

### Test a Model on New Images

#### 1. Choose at least five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web (by using Google street view in Berlin and Munich):

![test images][test_images] 

The first image might be difficult to classify because it is quite dark and the sign is relatively small. The second image could be difficult if the angle of the arrow differs too much from training data. The third image could be difficult because the pedistrian icons are small and blury. The fourth image could be difficult if the angle, position and size are not similar enough to the training data.
The same could be said for the fifth and sixth images, which have the additional problem of being  quite dark.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No entry      		| Speed limit (80km/h)   									| 
| Keep right     			| Speed limit (80km/h) 										|
| Children crossing					| Speed limit (80km/h)											|
| Yield	      		| Speed limit (80km/h)					 				|
| Stop		| Speed limit (80km/h)      							|
| General caution		| Speed limit (80km/h)      							|


The model was able to correctly guess 0 of the 6 traffic signs, which gives an accuracy of 0%. This does not compare well with the 94% accuracy on the test set.
An earlier version of the model correcly predicted 2 of the 6 traffic signs, but it had less than 93% validation accuracy, and so was not eligible for submission. In either case, the accuracy on web images is much, much lower than on the test data.

Clearly the model does not generalize well, which suggests I may be overfitting, or perhaps there is not enough variation in the training dataset. Augmenting the data set would be the next logical step.



#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


As discussed in the previous section, my model did not generalize well to the images found on the web. It is not very confident about its predictions, and when it is not confident, it typically thinks that "Speed limit (80km/h)" is the best choice.
The top five predicted traffic signs are similar for each image.

For the first image, the top five soft max values were

| Soft-max value         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.3         			| Speed limit (80km/h)   									| 
| 0.16     				| Priority road 										|
| 0.13					| No entry											|
| 0.09	      			| Speed limit (60km/h)					 				|
| 0.07				    | Beware of ice/snow      							|


For the second image,  

| Soft-max value         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.4         			| Speed limit (80km/h)   									| 
| 0.11     				| Speed limit (60km/h)										|
| 0.10					| Priority road											|
| 0.08	      			| Beware of ice/snow					 				|
| 0.07				    | No entry      							|


For the third image,  

| Soft-max value         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.36         			| Speed limit (80km/h)   									| 
| 0.16     				| Priority road 										|
| 0.12					| Speed limit (60km/h)											|
| 0.09	      			| Beware of ice/snow					 				|
| 0.04				    | Wild animals crossing      							|


For the fourth image,  

| Soft-max value         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.39         			| Speed limit (80km/h)   									| 
| 0.15     				| Priority road 										|
| 0.13					| Speed limit (60km/h)											|
| 0.06	      			| Beware of ice/snow					 				|
| 0.04				    | No entry      							|


For the fifth image,  

| Soft-max value         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.37         			| Speed limit (80km/h)   									| 
| 0.16     				| Priority road 										|
| 0.11					| Speed limit (60km/h)											|
| 0.07	      			| Beware of ice/snow					 				|
| 0.07				    | No entry      							|

For the sixth image,  


| Soft-max value         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.33         			| Speed limit (80km/h)   									| 
| 0.15     				| Priority road 										|
| 0.10					| Speed limit (60km/h)											|
| 0.10	      			| Beware of ice/snow					 				|
| 0.06				    | No entry      							|




