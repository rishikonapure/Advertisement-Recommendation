This Directory contains the dataset used for the project
# Data Collection 

## Text Data
To perform the text classification, YouTube videos metadata using YouTube API v3 is collected. This
dataset consists of fields: video id, title, description, hashtags, duration and upload details etc. A total
of more than 10,000 video metadata is collected and divided into six categories. This dataset items
contain a lot of noise which proves to be challenging. Before building a classification model we
should perform data pre-processing. Natural Language Toolkit (NLTK) package of NLP does the job
of pre-processing efficiently. It consists of data pre-processing tools required for text classification.

## Image Data
For the video classification model a Convolutional Neural Network is trained to identify the sport in
the video. Image data from Google images of 22 categories of sports (badminton, baseball, cricket,
football, etc.) is collected. Each sport category has around 700 – 800 images. The same dataset is used
to create a video classification model with Keras and deep learning. A comparison was done between
the developed model and the inception v3 model.

[Download the data from here.](https://www.kaggle.com/rishikeshkonapure/sports-image-dataset)

## Advertisement data
 In order to achieve this various advertisements in the video format are collected and metadata of these are extracted and stored in a database. The advertisement video data consists of information regarding video URL, video id, title, description etc.
