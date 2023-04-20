# Style Transfer Network for Video

This project demonstrates the use of a style transfer network to apply a given style to a video. The video is processed frame by frame, and the style is transferred using the `train_image` function. The processed frames are then merged back together to create the final stylized video.

## Overview

The project uses a pretrained VGG16 network for style transfer. The input video is captured frame by frame, and each frame is converted to a PIL image. The style is then transferred to each frame using the `train_image` function. Finally, the processed frames are merged back together to create the final stylized video.

## Style Transfer Neural Network Script Explanation


This code is an implementation of the Neural Style Transfer algorithm, which is used to transfer the style of one image (the style image) onto the content of another image (the original image). The result is a new image that combines the content of the original image with the artistic style of the style image.

### Key Components

- `FeatureExt`: A PyTorch module that uses the pretrained VGG-16 model as a feature extractor. It extracts features from selected layers of the VGG-16 model.
- `image_transform`: A set of image transformations applied to the input images, including normalization using the mean and standard deviation of ImageNet dataset.
- `get_image`: A function (not shown in the provided code) that loads an image and applies the `image_transform`.
- `encoder`: An instance of the `FeatureExt` module, used to extract features from the original, style, and generated images.
- `optimizer`: An instance of the `torch.optim.Adam` optimizer with a learning rate of 0.1 and betas [0.4, 0.99], used to optimize the generated image.
- `Content_weight` and `Style_weight`: Weights for the content loss and style loss, with values 1 and 150, respectively.

### Workflow

1. Load the original and style images, apply the `image_transform`, and create an initial generated image (a clone of the original image).
2. Set the generated image to require gradient, making it learnable.
3. Initialize the optimizer with the generated image as its parameter.
4. Set the number of epochs for optimization (100 in the provided code).
5. Initialize the `encoder` and freeze its parameters to prevent training the feature extractor.
6. For each epoch:
    - Extract features from the original, style, and generated images using the `encoder`.
    - Calculate the content loss as the mean squared error between the original and generated image features from the last selected layer.
    - Calculate the style loss as the sum of the mean squared error between the gram matrices of the generated and style image features, normalized by the product of the channels, height, and width of the feature maps.
    - Calculate the total loss as the weighted sum of the content and style losses, using `Content_weight` and `Style_weight`.
    - Perform a backward pass to calculate gradients and update the generated image using the optimizer.
    - Print the content and style losses for every 50th epoch.
7. Denormalize the final generated image and return it.

The code effectively performs Neural Style Transfer, generating a new image that combines the content of the original image with the artistic style of the style image.





  ## Dependencies

The following libraries are required to run this project:

 * torch

 * torchvision

 * PIL (Pillow)

 * numpy

 * matplotlib

 * cv2 (OpenCV)


 ## Usage

Clone the repository and install the required dependencies.
Provide an input video and a style image.
Run the main script to apply the style to the input video and create the final stylized video.
