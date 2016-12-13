**Image Classification**

The objective of this project is to build a Image classifier that
identifies the correct orientation of the image.

**About the dataset**
The dataset consists of micro-thumbnail images (8x8 pixels) represented
as 192 dimensional feature vectors (8x8x3, with each pixel represented
as a separate feature for Red, green and blue color densities).
The training datafile has one row per image, with each row formatted as:
photo_id correct_orientation r11 g11 b11 r12 g12 b12 ...
The testset has similar format except for the first 2 columns

**About the code**

The orient.py module has the main program that accepts arguments for path
to the training dataset and test datasets, algorithm (nnet)

The program can be run as below:

`./orient.py train-data.txt test-data.txt nnet
`
This invokes the Neural Network algorithm implementation in nnet.py

The neural network has one hidden layer. Implemented Cross-validation
based model selection to train the model  with different settings for
the hidden_layer_nodes parameter
Best parameter is then used and the model trained again on the full
dataset, that can be used to predict on unseen datasets.


**A brief Report**

Getting accuracy on an average about 70-75% for different configurations
of the nodes in the hiddlen layer.
Observing very small improvements in accuracy for NNs
with higher number of nodes but seems to mostly plateau for anything
with 16 and above nodes in the hiddel layer
The training is also pretty fast as well as the classification is fast
with NNs. Seems to be a good algorithm for predicting with the given
dataset, as the accuracy and computation speed seems to be pretty good

