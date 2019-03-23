# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[one_sign_per_class]: ./writeup_figures/one_sign_per_class.png "One Sign per Class"
[class_distribution]: ./writeup_figures/class_distribution.png "Class Distribution"
[before_augmentation]: ./writeup_figures/before_augmentation.png "Before Augmentation"
[after_augmentation]: ./writeup_figures/after_augmentation.png "After Augmentation"
[learning_curves]: ./writeup_figures/learning_curves.png "Learning Curves"
[internet_images]: ./writeup_figures/internet_images.png "Internet images"
[top_5_classifications]: ./writeup_figures/top_5_classifications.png "Top 5 classifications"

[misclassified_70_km_layers_conv1]: ./writeup_figures/misclassified_70_km_layers_conv1.png "Misclassified 70 km Layers Conv1"
[misclassified_70_km_layers_conv2]: ./writeup_figures/misclassified_70_km_layers_conv2.png "Misclassified 70 km Layers Conv2"
[misclassified_70_km_layers_conv3]: ./writeup_figures/misclassified_70_km_layers_conv3.png "Misclassified 70 km Layers Conv3"
[misclassified_70_km_layers_conv4]: ./writeup_figures/misclassified_70_km_layers_conv4.png "Misclassified 70 km Layers Conv4"

[test_visualization_conv1]: ./writeup_figures/test_visualization_conv1.png "Test Visualization Conv1"
[test_visualization_conv2]: ./writeup_figures/test_visualization_conv2.png "Test Visualization Conv2"
[test_visualization_conv3]: ./writeup_figures/test_visualization_conv3.png "Test Visualization Conv3"
[test_visualization_conv4]: ./writeup_figures/test_visualization_conv4.png "Test Visualization Conv4"

[example_misclassified_images]: ./writeup_figures/example_misclassified_images.png "Example Misclassified Images"

## Rubric Points
### Here we consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how each point was addressed.  

---
### Writeup / README

You're reading it! and here is a link to the [project code](https://github.com/koulakis/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The pandas library was used to calculate summary statistics of the traffic
signs data set:

* The size of training set is `34799`
* The size of the validation set is `4410`
* The size of test set is `12630`
* The shape of a traffic sign image is `(32, 32, 3)`
* The number of unique classes/labels in the data set is `43`

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. In order to better understand the nature of the data, we select one candidate from each class and plot the corresponding sign.

![alt text][one_sign_per_class]

Additionally it is important to know if there is some major balance issue between classes. We see that this is not the case by plotting the frequencies of the classes:

![alt_text][class_distribution]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The preprocessing consists of the following steps:
1. **Scaling to [0, 1]**:  This scaling is applied to all images just for them to have the right format for the augmentation functions.
1. **Data augmentation**: This step is performed during training and for each sample. The need for augmentation came from an error analysis where it was evident that most of the misclassified test images where the ones which had high or low brightness, high contrast or low saturation. Some examples are signs which are very bright when they are under the sun and ones which are under shadow. The augmentation reproduces those harder to resolve cases in all classes which makes the classifier more invariant under those transforms. Here is a list of them:
    - Brightness: We randomly add or remove brightness up to a factor of 0.3
    - Saturation: We randomly lower saturation by multiplying with a factor in the interval [0, 1]
    - Contrast: We randomly increase contrast by multiplying with a factor in the interval [1, 2]
1. **Scaling**: After applying augmentation, we scale around the mean value of all 3 color channels and in an interval of length 1

Here is an example of a sample and 10 random applications of the image transforms:

![alt_text][before_augmentation]
![alt_text][after_augmentation]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
|						|												|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x20 	|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x20 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x20 				|
|						|												|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 16x16x50 	|
| RELU					|												|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 16x16x50 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x50 				    |
| Flatten				| input: 8x8x50, output: 32000					|
| Dropout				| 0.8   										|
|						|												|
| Fully connected		| input: 32000, output: 500				        |
| RELU					|												|
| Dropout				| 0.8   										|
| Fully connected		| input: 500, output: 43         				|
| Softmax				|           									|
 
This model is a simple extension of LeNet. While visualizing the hidden layers of the original LeNet model, it was clear that the features learned in the convolution steps had too little flexibility. Therefore adding a second convolution layer on top of the existing ones, helped learn more complex non-linear features which increased the capacity of the model enough to over-fit on the training data set. After that, applying a dropout of 0.8 decreased over-fitting and allowed the model to perform well on the test and validation data sets.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
In order to monitor performance during training, we plot the learning curves of the training and testing data sets' accuracy. Here is a picture of it:

![alt_text][learning_curves]

Here is a description of the different choices made for the training setup:
- **Optimizer**: One of the requirements of the optimizer was to take momentum in account in order for convergence to be fast and not get stuck on local oscillations. On the other hand, having to manually set both the momentum and learning rate hyper parameters was too much overhead, thus an adaptive optimizer would be a good choice to start with. The two options then were `Adam` and `RMSProp`. Given that `Adam` is the most common choice for CNNs, this was the initial choice. There were no real difficulties in convergence during training, thus there was no reason to switch.
- **Batch size**: A small batch size is usually desired in order to take advantage of its regularization effect and also make sure that there is no chance to exceed the memory of the GPU (given the small image sizes, this would not be a problem). On the other hand having on average at least one candidate per class helps accelerate learning, thus 128 was a rough first choice. After trying out a couple of different batch sizes around this value, showed that it was a good choice.
- **Number of epochs**: The experiment started with a relatively large number of epochs. Then using the learning rate curve it was easy to spot where improvement stopped and use this information to reduce the number of epochs. There was never a performance decrease problem so no need for early stopping.
- **Learning rate**: During the first steps of the experiment the learning rate was adjusted after changes in regularization and batch size. Once those parameters were fixed, it was frozen to 0.001. After 20 or so epochs though the performance increase almost stopped and the learning curve was oscillating quite a bit. Therefore a decay of the learning rate was put in place and after 20 epochs it dropped to 0.0005. This smoothed the last part of the learning curve and also allowed to consistently achieve accuracy scores above 0.98.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
- training set accuracy of `1.0`
- validation set accuracy of `0.9845`
- test set accuracy of `0.9918`

An iterative approach was chosen and the model was built almost from scratch. This choice was made for learning purposes. It was fun building the model from scratch and seeing the whole approach approximating the functionality of keras, livelossplot and other higher level libraries of this sort. In a real-world scenario one would like to avoid this and instead reuse code, pre-trained models and utilize auto-ml as much as possible. 

Here is a timeline of the model development:
- **Initial architecture**: The first architecture chosen was that of LeNet because it is one of the simplest ones and the code for it was already available from previous courses.
- **Deficiencies of first approach**: It was evident that the model had too small capacity. 
- **Increase of capacity**: The capacity of the model was increased by doubling the convolution layers.
- **Overfitting**: The next problem was overfitting on the training dataset. This was resolve by using dropout after the flattening the output of the convolution part and also after the dense layer.
- **Data augmentation**: Looking at the misclassified images it was evident that data augmentation would help. The transforms added changed brightness, contras and saturation.
- **Learing rate decay**: The learning curve was oscillating a lot and did not smoothly converge to a solution. A decay of the learning rate after 20 epochs fixed this problem.

Here are some observations,concerns or next steps:
- **Overfitting of validation set**: Though the validation set was not involved in any decisions made during hyperparameter tuning (was only taken into consideration after 3-4 major milestones), it seems to be doing much better than the testing set. This could be an indication that some of the images of the training set are duplicated and exist on the validation set. No check has been made to verify this.
- **Augmentation**: The augmentation of images had a large effect in both performance improvement and generalization of the model. It would be nice though to try and tweak the model architecture to make it more invariant under the augmentation transforms instead of just keep changing the input data.
- **ML pipeline**: The whole pipeline is a mess. It spills around several cells, preprocessing is performed before and during training and the outputs of experiment are not versioned properly. The improvement of those parts would be essential to make life easier in case one where to continue working on & improving this classifier.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt_text][internet_images]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 50 km/h      		    | Entrance forbidden							| 
| Stop      			| Stop 									    	|
| Yield					| Yield											|
| Constructions	      	| Constructions					 				|
| 70 km/h   			| Constructions      							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This is much worse than the accuracy of the test and validation data sets. The main reason is probably that given the original data set, the model does not have to deal with vertical scaling or horizontal translations. If one where to add those transforms to the data augmentation, then this would improve those predictions. In order to verify this though, one should gather new data or extend the test and validation data set with appropriate augmentations. This way we could verify that vertical scaling and translation are deficiencies of the model and afterwards monitor improvement from a new pipeline tuning.

Looking at the outputs of the convolution layers, one can see for example in the 70 km sign which is shifted on the left, how it mashes with the background on the right and results to a wrong prediction:

![alt_text][misclassified_70_km_layers_conv1]
![alt_text][misclassified_70_km_layers_conv2]
![alt_text][misclassified_70_km_layers_conv3]
![alt_text][misclassified_70_km_layers_conv4]

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Here are the probabilities of the first choices made for each image:

| Image | Probability         	|     Prediction	        					| 
|:-----:|:---------------------:|:---------------------------------------------:| 
| 1     | 0.96         			| Entrance forbidden  							| 
| 2     | 0.99     				| Stop 											|
| 3     | 1.					| Yield											|
| 4     | 1.	      			| Constructions				 				    |
| 5     | 0.42				    | Constructions     							|

Thus the classifier seems to decide with high confidence in the three correct cases but also on the first wrong one.

Here is an image of the top 5 predicted images for each one of the 5 web images (confidence decreases from left to right):

![alt_text][top_5_classifications]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Here is an example output of the 4 neural network convolution layers:

![alt_text][test_visualization_conv1]
![alt_text][test_visualization_conv2]
![alt_text][test_visualization_conv3]
![alt_text][test_visualization_conv4]

We can see that the neural network is using basic geometrical shapes from the inside and the boarder of the signs in order to make predictions. It also uses colors to distinguish between the different parts of the sign.

### Debugging
A final section was added which was used to debug the model and drive decisions on where to focus at each step to make the next improvements. The debugging consists of two parts:
- Plotting 20 randomly picked misclassified images from the test set
- Plotting the intermediate layers of selected misclassified images (usually the ones having properties which appeared the most in the 20 errors)

The most significant contributions of this analysis were:
- Spotting the low capacity of the LeNet single convolution layers (intermediate layers)
- Identifying that very bright or dark images with low saturation produced the majority of misclassifications (plot of 20 random errors)

Looking at the final set of misclassified images, one can notice that the ones left are the most horrible ones. They have sunlight from the back, they are rotated, blurred from camera movement, have spots or are very dark. Here are some samples:

![alt_text][example_misclassified_images]