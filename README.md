# Classifying Satellite Imagery using Planet API and Neural Networks
## Cameron Bronstein
### Capstone Project - General Assembly Data Science Immersive

### Executive Summary
This project was a multi-label computer vision classification problem. I used machine learning to classify satellite images of the amazon into varying weather and land-use categories. Modeling included building a convolutional neural network to train and process a dataset that included approximately 40,000 training and 60,000 testing images.

#### Problem Statement
With several global leaders in satellite imagery now emerging and machine learning technology growing more advanced, earth imagery is entering a new and exciting period. From monitoring defense to monitoring earth's ecosytems, leveraging satellite imagery for analytical purposes is increasingly pressing. 

#### Gathering the Data
The data for this project is from this 2017 Kaggle competition, sponsored by Planet:
[Planet: Understanding the Amazon from Space.](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data)

However, to demonstrate generating my own images with Planet's API, I constructed a pipeline to pull and download geotiff files given an area of interest (AOI).

Given time constraints, I decided to move forward with the Kaggle Dataset. However, with increased permissions, it is highly feasible to use the Planet API to construct your own dataset for image analysis.

#### Exploratory Data Analysis and Preprocessing
EDA mainly included understanding the distributions of varrying label classes and generating images from the provided geotiff files. Plotting geotiffs requires altering their RGB-IFR values to create a better visual product. The tifs are originally in a format generated from "what the satellite sees".

There were also varying hurdles to overcome the enormaty of the dataset when preparing the images for modeling (due to RAM limits on my local machine). Geotiffs must be converted through a library called rasterio, that generated multi-dimensional arrays from the original files. 

Following preprocessing, the training and testing arrays were uploaded to AWS for cloud GPU support.

#### Modeling

In process...

Initial Plans:
- Build a "from-scrath" Convolutional Neural Net to predict the 17 classes using Keras and Tensorflow.

Time restricted: 
- Use an "image-net" pre-trained model for a much more robust prediction.
- Ensemble multiple models and use ridge regression to pick final labels.




