from model_train import train_image
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np

style_image = Image.open("gogh.jpg")

video = "test_clip.mp4"
output = "result.mp4"

# Read the input video
cap = cv2.VideoCapture(video)

# Get the video's FPS
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize the VideoWriter object
out = None

# Process each frame of the video
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the cv2 image (BGR) to a PIL image (RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame)

    print(f"frame {frame_count} /{fps} is editing")
    frame_count += 1

    # Process the PIL image with the train_image function
    processed_pil_image = train_image(pil_image, style_image)

    # Convert the processed PIL image back to a cv2 image (BGR)
    processed_frame = cv2.cvtColor(np.array(processed_pil_image), cv2.COLOR_RGB2BGR)

    # If the VideoWriter object is not initialized, do it based on the first processed frame
    if out is None:
        height, width, _ = processed_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output, fourcc, fps, (width, height))

    # Write the processed frame to the output video
    out.write(processed_frame)

# Release the video objects
cap.release()
if out is not None:
    out.release()








