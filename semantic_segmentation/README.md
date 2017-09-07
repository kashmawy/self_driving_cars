# Semantic Segmentation
### Introduction

In this project, we label the pixels of the road in images using a Fully Convolution Network.

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Run
Run the following command to run the project:
```
python3 main.py
```

### Code Parts

1. Load VGG
2. Create a fully convolution network from VGG by doing the following changes:
  a. Connecting vgg layer7 to a 1x1 convolution and then to a transposed convulational network with a 4x2 output
  b. Connecting vgg layer4 to a 1x1 convolution to serve as a skip layer along with the output from part a together into a transposed convolution network to give a 4x2 output
  c. connecting vgg layer3 to a 1x1 convolution to serve as a skip layer along with the output from part b together into a transposed convolution network to give a 16x8 output
  The output of c is the final output, where it will give an image of the size 16x8
3. Create a placeholder tensor to hold the learning_rate and another to hold correct_label with dimensions (None, None, None, num_classes) where num_classes are 2 [road or no_road]
4. Optimize the network by taking the output of c as logits and reshape them to (-1, num_classes), and take the correct_label and reshape it to (-1, num_classes).
   Sets the loss function to be cross entropy softmax of logits and labels and uses Adam Optimizer in order to reduce it
5. Train the network by doing the following:
  a. By going epochs time (epoch = 10)
  b. For each time, grabbing a batch of 10 and run the session with input_image being the image, learning_rate of 0.0005, keep probability for dropout of 0.7, and correct label.
     As we train we print the epoch and training loss
6. Save the model after training
7. Run the model on testing images and produce the result in the run folder

The following shows couple of images after inference:

![Semantic Segmantion Inference Example 1](runs/1504767227.3010378/uu_000039.png)

![Semantic Segmantion Inference Example 2](runs/1504767227.3010378/uu_000096.png)