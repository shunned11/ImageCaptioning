# Introduction
A model that suggests a suitable caption based on the image using the concepts of Natural Language Processing and Computer Vision

# Dependencies
* Python 3.7.7
* Tensorflow 2.2.0
* Pickle5
* Numpy 1.18..5
* Pandas 1.0.3
* Matplotlib 3.1.3

# Required Data
Both raw data and extracted data in the form of .npy files for direct use can be found on the given link

# Model Description
The input image is passed through a pre-trained CNN(in this case the Xception Model pretrained on the ImageNet dataset is used) followed by a dropout layer and then a fully connected layer. The partial caption is passed through an embedding layer(GloVe Vector in this case) followed by two layers of a dropout layer and a bidirectional LSTM.

The ouptut of both the LSTM layers(in case of acption model) and dense layer(in case of image model) are then merged to ouput a fully connected layer
