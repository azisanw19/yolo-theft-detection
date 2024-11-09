# Mobile Net Theft Detection

## Steps

The dataset used in this article is the thief and guest dataset. First, the image data of thieves and guests is collected.

After collecting the thief and guest datasets, the image will be segmented first before conducting training. Segmentation is performed automatically using a mobile net model that has been trained.

The mobile net will provide a boundary person for the image, which will conduct training with the aim of classifying guests and thieves as well as obtaining a boundary person for the image.

## Requirements

- [keras](https://keras.io/)
- [opencv-python](https://opencv.org/)
- [numpy](https://numpy.org/)
- [tensorflow](https://www.tensorflow.org/?gad=1&gclid=Cj0KCQjwy9-kBhCHARIsAHpBjHiVq746_swfKE74YN9gLldJ8-Y7uhK88Vc1p8rNtNAxmaSstKRgTv8aAhvwEALw_wcB)
- [pandas](https://pandas.pydata.org/)

## Generate Segmentation using Mobile net

MobileNet is a simple but efficient and not very computationally intensive convolutional neural networks for mobile vision applications. MobileNet is widely used in many real-world applications which includes object detection, fine-grained classifications, face attributes, and localization.

## Classification theft pre-trained model mobile net

The training was conducted using a pre-trained mobile net model. The classification model used can be seen below:

![Classification pre-trained Mobile net](/result/model-classification.png)

## Implementation

Save what has been trained, then use it in live video broadcasts.
