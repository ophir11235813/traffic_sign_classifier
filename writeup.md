# Classifying traffic signs using deep convolutional neural networks 

In this project, I've created a classification model for (German) road traffic signs, using a deep convolutional neural network built off a variation on the LeNet architecture. I am also pre-processing images to improve accuracy. The below code takes ~10 minutes to train on a MacBook Pro (2015) over ~20 epochs, and <b> achieves 96% accuracy on the test set. </b>

The goals / steps of this project are the following:
1. Explore, summarize and visualize the data set
2. Pre-process the images and augment the dataset
3. Design, train and test, a model architecture
4. Use the model to make predictions on new images and analyze their softmax probabilities


---



## 1. Explore the dataset

The (pickled) training, validation, and testing data is here. In summary, it has 51,839 samples which are broken into:
<ul> 
<li> 34,799 training samples (67%) </li>
<li> 4,410 validation samples (9%) </li>
<li> 12,630 testing samples (24%) </li>
</ul>

Each image's dimensions are 32x32x3, where the third dimension represents the three color channels of the (color) image. There are 43 separate classes of image (i.e. types of road signs), and below is a representative sample from each class:

[//]: # (Image References)

![image1](https://raw.github.com/ophir11235813/traffic_sign_classifier/master/images/Traffic_signs_full.png)

Note that the training data is <i> not </i> evenly distributed amongst the classes: 

![image2](https://raw.github.com/ophir11235813/traffic_sign_classifier/master/images/histogram_before.png)

## 2. Pre-process the images and augment the dataset

I will pre-process each image of the training, validation, and test data to improve the accuracy of the model. This pre-processing takes advantage of knowledge we have about the traffic signs:

<ul>
<li> <b> Focus on region of interest: </b> Each image comes with coordinates of the (two opposite corners of the) bounding box around the sign. I first crop the image to that bounding box, and then resize the image so that it is of size 32x32x3. </li>
<li> <b> Convert to grayscale: </b> The images' color is not relevant to their meaning, and hence we can convert to grayscale. This reduces the color channels from three to one. </li>
<li> <b> Apply adaptive histogram equalization: </b> This improves the contrast in the images, and enhances the definitions of edges.
<li> <b> Normalization: </b> The values of the channels are then transposed by 0.5, so that they are all in the interval [-0.5, 0.5] </li>
</ul>

Here are two examples of images before (left) and after (right) pre-processing: 
![image3](https://raw.github.com/ophir11235813/traffic_sign_classifier/master/images/processing.jpg)


As the dataset has very few samples of some classes (see the above histogram), I will next augment the dataset by <i> creating</i> more examples of the under-represented classes. I do this by rotating each image from -25 to +25 degrees from the original, and then concatenating (augmenting) the dataset to include the new images. 

See appendix A for a summary of the number of images I generate/add to each class. The resulting training dataset has 89,741 rows, distributed over the classes as follows:

![image4](https://raw.github.com/ophir11235813/traffic_sign_classifier/master/images/histogram_after.png)

## 3. Design, train, and test the neural network

The convolutional neural network (CNN) used for this model is my modification of the <a href="http://yann.lecun.com/exdb/lenet/"> LeNet </a>, first developed by Prof. Yann LeCun, published in 1994. I chose this architecture as it performs well on character and number recognition (key features of the traffic signs), while also being computationally light enough that I can run it on my MacBook Pro (2015) without GPU support. It accepts 32x32x1 images, and computes the logits over five layers, implementing max-pooling, convolutions, and dropout. Here is the model architecture:

<ul>
<li> <b> Layer 1</b>: Convolutional layer (valid padding with single stride size), then activated with Relu. This takes dimensions from 32x32x1 to 28x28x6. Then apply max-pooling, with stride width/height = 2, taking dimensions from 28x28x6 to 14x14x6. Apply droupout. </li>
<li> <b> Layer 2</b>: Convolutional layer (valid padding with single stride size), then activated with Relu. This takes dimensions from 14x14x6 to 10x10x16. Then apply max-pooling, with stride width/height = 2, taking dimensions from 14x14x6 to 5x5x16. Then apply dropout. Flatten this to a single vector of size 1x400 </li>
<li> <b> Layer 3</b>: Fully connected layer: Activate with Relu, and include a dropout. Input size = 400, output size 120 </li>
<li> <b> Layer 4</b>: Fully conneted layer: Activate with Relu. Input size = 120, output size 84 </li>
<li> <b> Layer 5</b>: Fully conneted layer: Input size = 84, output size 43 (the number of classes) </li>
</ul>

See appendix B for a summary of the model. 

To train the model, I fed the (pre-processed and augmented) training data into the above CNN in batches of size 128 rows. For each row in each batch, I computed the accuracy by comparing the model's output (logits, or probabilities) to the truth (one-hot) vector. I repeated this for each batch, and then computed the total accuracy as the average accuracy over all the batches. 

The above process defines one epoch, or iteration, of the model's training. For each epoch, we optimize the parameters (i.e. the values of the weights matrices) by minimizing the loss function. Here, the loss function is the cross-entropy of the standard softmax probabilities, and the optimization method is the <b> Adam optimizer </b> which is a first-order gradient-based method, based on adaptive estimates of lower-order moments. I use a static learning rate of 0.001. More details about the optimizer can be found <a href = "https://arxiv.org/abs/1412.6980">here</a>.

In summary, the model is:
<ul>
<li> A modified LeNet 5-layer convolutional neural network, with input dimensions 32x32x1 and output dimension 43 (number of classes) </li>
<li> Optimized using Adam, with learning rate = 0.001 </li>
<li> Epochs = 30 </li>
<li> Batch size = 128 </li>
</ul>

<b> A note on my approach: </b> Implementing the standard LeNet architecture on unprocessed images resulted in an approximately 87% test accuracy. I experimented with various methods to improve the accuracy, some of which were effective while others were not. For example, normalizing the images using the formula (image_value - 128)/128 or <i>only</i> focusing on the bounding box did not improve accuracy. However, converting the image to grayscale, strengthening the definition of its color boundaries (through adaptive histogram equalization), and applying dropout did improve accuracy substantially above 87%.

The training and validation accuracies gradually grew from 80% and 85% respectively, to the following final results:
* training set accuracy of 99.7%
* validation set accuracy of 96.9% 
* test set accuracy of ?

Below is a graph of how my training and validation accuracies improve over the 30 epochs. 

![image5](https://raw.github.com/ophir11235813/traffic_sign_classifier/master/images/accuracy_epochs.png)

## 4. Using the model to classify new images

Once trained, the above model can be used to classify the following five new traffic signs (downloaded from Google Images): 

![image6](https://raw.github.com/ophir11235813/traffic_sign_classifier/master/images/five_signsb.png)

While the first image (stop sign) is similar to training images, the other images are slightly more challenging than those the model trained on: the third is tilted, the forth has two contrasting colors in the background, and the fifth is partially obstruted by a pole. The second image (elderly crossing) <i> isn't in the training data </i> but is similar to the children crossing sign. 

The model accurately classified all the images except for the second (hence achieving an 80% accuracy). It classified the second image (elderly crossing) as a children crossing. Here are the results of the prediction:

| Image			        |     Prediction	        					| Correct? |
|:---------------------:|:---------------------------------------------:|:---------:|
| No Entry					| No Entry											| Yes| 
| Elderly crossing      		| Children crossing   									|  No! |
| Road work     			| Road work 										| Yes |
| 120 kph limit	      		| 120 kph limit				 				| Yes| 
| Pedestrians		| Pedestrians      							|Yes|

Consider the fourth image (120 kph), which the model correctly classifies (softmax probability 70%). The next four most "likely" predictions are <i> also </i> speed signs, of limits 30 kph (29%), 100 kph, 80 kph, and 50 kph (the last threes' probabilities add up to <1%). 

![image7](https://raw.github.com/ophir11235813/traffic_sign_classifier/master/images/german_results.png)



| Sign         	|     Top five softmax probabilities        					| 
|:---------------------:|:---------------------------------------------:| 
| No Entry         			| <b>No Entry (>99%)</b>, all others <1% (Stop, Turn left, No Passing, Priority road)	| 
| Elderly crossing			| <b>Children crossing (>99%)</b>, all others <1% (Bicycles, Road narrows on right, Road work, Dangerous curve on right) 										|
| Road work					| <b>Road work (97%)</b>, Children crossing (2.5%), Beware of ice (0.3%), <0.2% (Bicycles crossing, double curve)										|
| 120 kmp limit	    | <b>120 kph	(70%)</b>, 30 kph (29%), <1% (100 kph, 80 kph, 50 kph)				 				|
| Pedestrians				    | <b>Pedestrians (99%)</b>, <1% (Childrens crossing, Dangerous curve to right, Right of way at intersection, Road narrows on right      							|








## Appendices:

### Appendix A: Number of augmented images per class

| Class         		|     Number of generated images	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0         		| 1907  							| 
| 1     	| 107	|
| 2					|		77	|
|3	      	| 827	|
| 3    | 317    									|
| 4		| 437       									|
| 5				| 1727		|
|	6				|				797								|
|	8				|						827						|
|	9				|									767			|
|	10				|					287							|
|	11				|								917				|
|	12			|				197								|
|	13			|							167					|
|	14			|					1397							|
|	15				|								1547				|
|	16				|						1727						|
|	17			|					1097							|
|	18			|						1007						|
|	19			|						1907						|
|	20				|				1787								|
|	21				|								1817				|
|	22			|			1757									|
|	23			|			1637									|
|	24			|				1847								|
|	25				|			737									|
|	26				|		1547										|
|	27			|				1877								|
|	28			|				1607								|
|	29			|				1847								|
|	20				|							1697					|
|	31				|											1397	|
|	32			|		1877										|
|	33			|			1488									|
|	34			|			1727									|
|	35				|			1007									|
|	36				|							1757					|
|	37			|				1907								|
|	38			|					227							|
|	39			|								1817				|
|	40				|			1787									|
|	41				|							1877					|
|	42			|							1877					|


### Appendix B: Architecture of the CNN

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| 2D convolution  	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Dropout  	| probabilities defined during implementation|
| 2D convolution  	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Dropout  	| probabilities defined during implementation|
| Flatten    | 400x1      									|
| Fully connected		| outputs 1x120        									|
| RELU					|												|
| Dropout  	| probabilities defined during implementation|
| Fully connected		| outputs 1x84       									|
| RELU					|												|
| Dropout  	| probabilities defined during implementation|
| Fully connected		| outputs 1x43       									|
