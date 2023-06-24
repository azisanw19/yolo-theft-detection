# Theft Detection

---

## Steps

The dataset used in this article is the thief and guest dataset. First, the image data of thieves and guests is collected.

After collecting the thief and guest datasets, the image will be segmented first before conducting training. Segmentation is performed automatically using a mobile net model that has been trained.

The mobile net will provide a boundary person for the image, which will conduct training with the aim of classifying guests and thieves as well as obtaining a boundary person for the image.

## Generate Segmentation using Mobile net

MobileNet is a simple but efficient and not very computationally intensive convolutional neural networks for mobile vision applications. MobileNet is widely used in many real-world applications which includes object detection, fine-grained classifications, face attributes, and localization.