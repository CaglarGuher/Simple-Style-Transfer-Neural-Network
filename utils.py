import torch
import numpy as np
from PIL import Image

device = torch.device("cuda")
def get_image(image,img_transform, size = (500,500)):

    image = image.resize(size,Image.LANCZOS)
    #"Lanczos" refers to the Lanczos resampling algorithm, a mathematical technique 
    # used for image resizing, also known as Lanczos resampling.
    image = img_transform(image).unsqueeze(0)
    #this is required because we have to add batch size into image
    #it will add 1 for the array(tensor)
    return image.to(device)

def get_gram(m):
    _,c,h,w = m.size()

    m = m.view(c,h*w)
    m =torch.mm(m,m.t())
    return m

def denormalize_img(x):
    x = x.numpy().transpose((1,2,0))
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225]) 
    #ImageNet images std and mean values
    #it will be useful because we will use transfer learning
    x = std*x+mean
    x = np.clip(x,0,1)
    pil_image = Image.fromarray(np.uint8(x * 255))
    return pil_image


