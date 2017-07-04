# Deep convolutional neural network, to classify traffic signs

In this project, I've created a classification model for common (German) road traffic signs, using a deep convolutional neural network built off a variation on the LeNet architecture. I am also pre-processing images to improve accuracy. The below code takes ~10 minutes to train on a MacBook Pro (2015) over ~20 epochs, and <b> achieves >96% accuracy on the test set. </b>

---

## High level project goals

The goals / steps of this project are the following:
1. Explore, summarize and visualize the data set
2. Pre-process the images and augment the dataset
2. Design, train and test, a model architecture
3. Use the model to make predictions on new images
4. Analyze the softmax probabilities of the new images

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
<li> <b> Normalization: </b> The values of the channels are then transposed by 0.5, so that they are all in the interval [-0.5, 0.5] <;li>
</ul>

Here are two examples of images before (left) and after (right) pre-processing: 
![image3](https://raw.github.com/ophir11235813/traffic_sign_classifier/master/images/processing.jpg)


As the dataset has very few samples of some classes (see the above histogram), I will next augment the dataset by <i> creating</i> more examples of the under-represented classes. I do this by rotating each image from -25 to +25 degrees from the original, and then concatenating (augmenting) the dataset to include the new images. 

See appendix A for a summary of the number of images I generate/add to each class. The resulting training dataset has 89,741 rows, distributed over the classes as follows:

![image4](https://raw.github.com/ophir11235813/traffic_sign_classifier/master/images/histogram_after.png)

## 3. Design, train, and test the neural network

The convolutional neural network (CNN) used for this model is my modification of the <a href="http://yann.lecun.com/exdb/lenet/"> LeNet </a>, first developed by Prof. Yann LeCun. It accepts 32x32x1 images, and computes the logits over five layers, implementing max-pooling, convolutions, and dropout. Here is the model architecture:

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













[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"



####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:


 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?



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
