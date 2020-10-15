# Image Segmentation

In this example, we show a basic image segmentation algorithm to partition an image into segments based on their pixel values.  To solve this problem, we use D-Wave's hybrid discrete quadratic model solver, and demonstrate how to build a DQM object from a set of numpy vectors.

Additionally, this repository demonstrates the ability of D-Wave's Leap IDE to automatically load a new workspace with specialized packages using a YAML file.

## Usage

To run the demo, type the command:

```python image_segmentation.py```

This will build a random image based on the specifications stated by the user.  The first prompt will ask for the dimensions in pixels (a square image will be created), and the second prompt will ask how many segments we want in our image.

Alternatively, the user can specify an input image such as ```test_2_segments.png``` by typing:

```python image_segmentation.py test_2_segments.png```

The user is then prompted for the number of segments we wish to partition the input image into.

After the program executes, a file is saved as ```output.png``` that shows the original image on the left and the partition outlines in an image on the right.

**Note**: For this demo to run relatively quickly, image sizes should be kept below 50x50 pixels.

## Building the DQM

A simple method to partition an image into segments is to compare their pixel values.  If colors are similar, then they might belong to the same object in the image. This program builds a DQM object in which we have a variable for each pixel and a case for each segment.  As we compare pixels, we examine their difference using the provided ```weight``` function, which assigns smaller values for more alike colors, and larger values for more different colors.  Using this weight function, we assign quadratic biases between pixels in the same cases.  As the solver minimizes the energy landscape, it is then minimizing the difference between pixels placed in the same segment or partition.

## Initializing the Leap IDE

By creating the file ```.gitpod.yml``` in this repository, we are instructing the Leap IDE to load a new workspace and run the corresponding tasks listed in the file.  In this case, the task is to install the packages indicated in the file ```requirements.txt```.  When a workspace is created from this repository, these packages will be automatically installed without any action from the user.
