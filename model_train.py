import torch
import torch.nn as nn
from torchvision import models , transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils import get_image , get_gram , denormalize_img

def train_image (Original_image,Style_image):

    device = torch.device("cuda")
    class FeatureExt(nn.Module):
        def __init__(self):
            super(FeatureExt, self).__init__()
            self.selected_layers = [3, 8, 15, 22]
            self.vgg = models.vgg16(pretrained=True).features
            
        def forward(self, x):
            layer_features = []
            for layer_number, layer in self.vgg._modules.items():
                x = layer(x)
                if int(layer_number) in self.selected_layers:
                    layer_features.append(x)
            return layer_features

    image_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean = (0.485,0.456,0.406),std = (0.229,0.224,0.225))])


    Original_image= get_image(Original_image,image_transform)

    Style_image = get_image(Style_image,image_transform)

    Resulted_image = Original_image.clone()
    #the reason is that initial image created will be original image with 
    #loss is 0 and add the style as we train(much faster than creating random parameter and then 
    # match with the style)
    Resulted_image.requires_grad
    # wee need to make it learnable so 
    Resulted_image.requires_grad = True
    Resulted_image.requires_grad
    optimizer = torch.optim.Adam([Resulted_image],lr = 0.1,betas=[0.4,0.99])
    Epoch = 100
    #learning rate and beta values are in the published article for Style Transfer
    encoder = FeatureExt().to(device)

    for p in encoder.parameters():
        p.requires_grad = False  #to freeze the layers for transfer learning


    Content_weight = 1
    Style_weight = 150  #These are the values published from the Style Network Transfer


    for epoch in range(Epoch):

        Original_features = encoder(Original_image)
        Style_features = encoder(Style_image)
        Final_features = encoder(Resulted_image)

        Content_Loss = torch.mean((Original_features[-1]-Final_features[-1])**2)
        #last element of selected layer(relu activated layer)

        Style_Loss = 0

        for x_x,y_y in zip(Final_features,Style_features):
            _,c,h,w = x_x.size()
            gram_gen_feature = get_gram(x_x)
            gram_style_feature = get_gram(y_y)
            Style_Loss  = Style_Loss + torch.mean((gram_gen_feature-gram_style_feature)**2)/(c*h*w)
            # we are dividing to c*h*w to normalize the loss else it might be too big)
        loss = Content_weight*Content_Loss + Style_weight*Style_Loss 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch} :  Content loss: {Content_Loss.item():.4f} Style loss : {Style_Loss.item():.4f}")

            final_frame = Resulted_image.detach().cpu().squeeze()
            final_frame = denormalize_img(final_frame)
    return final_frame
    
