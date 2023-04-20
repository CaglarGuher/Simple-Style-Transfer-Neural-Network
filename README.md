# Style Transfer Network for Video

This project demonstrates the use of a style transfer network to apply a given style to a video. The video is processed frame by frame, and the style is transferred using the `train_image` function. The processed frames are then merged back together to create the final stylized video.

## Overview

The project uses a pretrained VGG16 network for style transfer. The input video is captured frame by frame, and each frame is converted to a PIL image. The style is then transferred to each frame using the `train_image` function. Finally, the processed frames are merged back together to create the final stylized video.

## Code Example

Here is an example of the `train_image` function:

```python
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils import get_image, get_gram, denormalize_img

def train_image(Original_image, Style_image):
    # ... (rest of the function code)
    final_frame = Resulted_image.detach().cpu().squeeze()
    final_frame = denormalize_img(final_frame)
    return final_frame
    
```


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