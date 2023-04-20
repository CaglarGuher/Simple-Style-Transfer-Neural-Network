Video Style Transfer
This project demonstrates a simple implementation of style transfer on videos using a pre-trained Style Transfer Network. The network is applied to each frame of the input video, generating a new video with the desired artistic style.

Overview
In this project, we use a style transfer network to transform the appearance of a video. The process involves capturing the video frame by frame, applying the style transfer using the train_image function, and then merging the stylized frames back together to create the final output video.

Style Transfer Network
Style transfer is a technique that allows the artistic style of one image to be combined with the content of another image. A pre-trained neural network is used to extract and transfer the style features from a given style image to the content of the input frames.

Processing Video Frames
The input video is read frame by frame using the OpenCV library. Each frame is then converted from BGR to RGB format and passed to the train_image function, which applies the desired style to the frame. After processing, the frame is converted back to BGR format.

Merging Stylized Frames
Once all the frames have been processed and stylized, they are merged back together into a video with the same frame rate as the original video. The final stylized video is saved as an MP4 file.

Usage
Place your input video (e.g., test_clip.mp4) and style image (e.g., gogh.jpg) in the project directory.

Edit the style_image and video variables in the script to match the filenames of your chosen style image and input video.

Run the script. The output video (e.g., result.mp4) will be saved in the project directory.

Dependencies
Python 3.6+
OpenCV
PyTorch
PIL (Python Imaging Library)
Conclusion
This project demonstrates a simple approach to applying style transfer to videos using a pre-trained style transfer network. By processing each frame individually and merging the stylized frames back together, we can create an artistic version of the original video.