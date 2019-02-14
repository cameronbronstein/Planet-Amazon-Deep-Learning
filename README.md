# Classifying Satellite Imagery using Planet API and Neural Networks
## Capstone Project - General Assembly Data Science Immersive
### Cameron Bronstein

### Executive Summary
This project is a multi-label computer vision classification problem. I used machine learning to classify satellite images of the amazon into varying weather and land-use categories. Modeling included building a convolutional neural network to train and process a dataset that included a 20 GB, 40,000 image training dataset and 30 GB - 60,000 image test dataset.


Sections (skip to...):
- [Problem Statement]([#Problem-Statement])
- [Gathering the Data](#Gathering-the-Data)
- [Exploratory Data Analysis](#Exploratory-Data-Analysis)
- [Moving to the Cloud with AWS](Moving-to-the-Cloud-with-AWS)
- [Image Pre-Processing: Test-Time Augementation](#Image-Pre-Processing:-Test-Time-Augementation-(TTA))
- [Modeling](#Modeling)
- [Results](#Results)
- Future Steps and Limitations(#Future-Steps-and-Limitations)


#### Problem Statement
With several global leaders in satellite imagery now emerging and machine learning technology growing more advanced, earth imagery is entering a new and exciting period. I am particularly interest in leveraging remote sensing for monitoring earth's ecosytems, and using new satellite imagery technology for conservation purposes is increasingly pressing. 

For this project, I built a neural network to tackle a multi-label classification problem.

#### Gathering the Data
The data for this project is from this 2017 Kaggle competition, sponsored by Planet:
[Planet: Understanding the Amazon from Space.](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data)

However, to demonstrate generating my own images with Planet's [API](https://planetlabs.github.io/planet-client-python/api/index.html), I constructed a pipeline to pull and download geotiff files given an area of interest (AOI). This was an exciting aspect of the project as I thoroughly enjoy data pipeline engineering tasks, and exploring the functionality of new APIs.

Given time constraints, I decided to move forward with the Kaggle Dataset. However, with increased permissions (i.e. access to data outside of Planet's free Open California imagery), it is highly feasible to use the Planet API to create more data for the sake of this project.

**From Kaggle Dataet**

This was my first forray into a more serious big-data project. The dataset included a ~20 GB training dataset (40,000 labeled images) and ~30 GB test dataset (unlabled images to be submitted and scored on competition website). Each image is approximately 525 KB.

The images come from Planet's Planetscope Satellites and are 3 M resolution.
Included are Analytic Geotiff and Visual JPEG formats - cropped to 256 x 256 pixels.
The Geotiffs contain four color bands (Red, Green, Blue, Near Infrared), while the JPEGs are correct into a 3 band visual product. 

#### Exploratory Data Analysis and Image Preprocessing
EDA mainly included understanding the distributions of varrying label classes and generating images from the provided geotiff files. Plotting geotiffs requires altering their RGB-IFR values to create a better visual product. The tiffs are originally in a format generated from "what the satellite sees".

There were also varying hurdles to overcome the enormaty of the dataset when preparing the images for modeling (due to RAM limits on my local machine). Geotiffs can be converted to numeric array (necessary for processing by the neural network) through a library called [rasterio](https://github.com/mapbox/rasterio).

I wrote a custom function that achieves this, however, significant RAM is needed to achieve this on a single machine.

#### Moving to the Cloud with AWS

Once I realized the data size and RAM hurdles, I changed gears and moved the project to the cloud using Amazon Web Serves Deep Learning GPU. This gave me new experience using big data tools offered in the cloud, and requires maneuvering of Elastic Block Storage Volumes to accomodate for my large dataset. 

However, I still faced memory limitations using the AWS servers, so I had to use Image Pre-Processing tools native to KERAS to prepare my dataset for modeling.

#### Image Pre-Processing: Test-Time Augementation (TTA)

I was introduced to `ImageDataGenerator` class in Keras Image Preprocessing. This allowed me to apply test-time augmentation (random flipping, rotating, zooming of training images) for better modeling approach.

The Image Data Generator pulls geotiff formatted images from SSD in batches, applies TTA, reads in the images and converts to numeric array, and runs them through the network. This greatly reduces memory load and allowed me to train the neural network and make pred on over 50 GB of images with on 7.5 GB RAM on AWS GPU server.

#### Modeling

I built a modeling pipeline that could accomodate for multiple models and cross validation. Unfortunately, bugs in the code prevented me from training the neural network on multiple cross validation folds (see Limitations and Future Steps section below). 

I trained a small Convolutional Neural Network using resized training images (downsized to 3-band, 128 x 128 pixels due to memory limitations). Network architecture was as follows: 

- Batch Normalization on input 
- 32 filter convolutional layer + Max Pooling Layer
- 64 filter convolution layer + Max Pooling Layers
- Flatten output - 128 Node Densely Connected Layer
- 17 Node output layer (representing 17 target label classes).

I had hope to train using deeper neural network (I set up functions for these models that could easily be passed into the code given higher computing power).

I trained my model over 25 epochs (20 epochs -- weights re-initialized -- 5 more epochs).

#### Results

I tracked Accuracy and F-Beta Score, but placed focused on F-Beta, as this is a more stringent classification metric compared to accuracy (especially for multi-label classification). F-beta is an average of Precision and Recall (Sensitivity) that includes a weighted penalty on false positives and false negatives, hence it's harsher scores compared to accuracy

Final F-Beta Score on the validation data was **0.843**. Kaggle submission unlabeled dataset was similar (0.834), showing my model generalizes well on a broad range of unseen data.

My model is slightly underfit, which could be attributable to overly aggressive TTA or excessive regularization in the network (droupout of 0.5 at multiple layers). 

Lastly, model predictions results in relative instability of the F-beta metrics on the validation set. Upon further investigation, it appears that the model was overpredicting categories in it's output. For the validation data (of which we could track predictions vs true labels), the model output > 43,000 labels for ~10,000 images, of which there were only ~28,000 true labels across the data. This would likely contribute to the instability of F-Beta Scores.

Model predictions were greatly impacted by the imbalanced classes. Predictions rarely included labels that had under 1,000 initial observations in the traiing data. While these images likely were of low representation in the test dataset, it is important that our model could distinguish and predict these features as well.

#### Future Steps and Limitations

**Limit number of label predictions per image or per dataset**
To correct the issue of over-predicting, I could limit the total number of possible labels when converting model outputs (probabilities) to better reflect the possible number of labels. What would this look like? First, only consider the top n probabilities. Of those, only accept those above the threshold value.

**Test F-beta score on predictions using multiple thresholds**
Another fix could be to more aggressively find appropriate thresholds for individual categories. Successfully implementing k-fold cross validation and training k models would allow for k searches for optimum thresholds. 

**Bootstrapping to Balance Label Classes**
I would also bootstrap (create "replicate" samples of infrequent labels) by target images with low sampled labels, augmenting those images and adding the augmented copies back into the dataset. This would likely improved training regardless of initial frequency of labels in the dataset.

**Invest in powerful computing resources**
The main limitation in my project was RAM. While having access to a remote GPU was vital to reduce training times, I ran into considerable hurdles throughout the project with overloading the RAM (despite the measures I took to reduce batch size). These memory challenges, along with out-of-pocket cost of using cloud-based servers, also made it difficult to experiment with deeper neural networks (I was especially excited about the potential of using ResNet 50 with "ImageNet" weights).

**Experiment with Model Ensembling**
I learned through research in Kaggle Discussions of the value of ensembling for neural netoworks. On a basic level, multiple CNNs could be trained and would likely make better predictions on certain labels in the dataset. The predictions from each neural network could then be combined, and, for each output (possible label), a linear regression model would be used to predict the "true value" of the label for each image. The different CNN predictions would be features to predict the target value (either 1 or 0, presence or absense). These new values would still be subject to the same thresholds used to finally land upon presence or absence of a label in each image.

**Consider Model Effectiveness and Resource Use**
The most effective modeling technique possbile would, essentially, require unlimited resources. While a complex, multi-model ensembling approach might yield the best predicting power, stakeholders must consider the cost (money and time) required to construct such an immense modeling system, as well as the reduced speed of classification if the models are to be used in real time. This is an ongoing, case by case trade-off, but one that must be considered in the context of deep learning.




