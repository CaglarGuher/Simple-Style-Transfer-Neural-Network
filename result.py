from model_train import train_image
import matplotlib.pyplot as plt
import cv2
from PIL import Image
img1 = Image.open("gogh.jpg")

img2 = Image.open("sky.jpg")
plt.imshow(img1)
plt.imshow(img2)

resulted_img = train_image(img2,img1)

output_image_path = "image.png"

resulted_img.save(output_image_path)