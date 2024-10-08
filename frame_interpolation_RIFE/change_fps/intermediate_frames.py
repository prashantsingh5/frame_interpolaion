import torch
import cv2
import numpy as np

import sys
import os

# Add the ECCV2022 directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'ECCV2022')))

from model.RIFE import RIFE  # Now it should work



# Load the RIFE model
model = RIFE(model_path=r"C:\Users\pytorch\Desktop\frame_interpolation\ECCV2022-RIFE-arxiv_v5_code\train_log\flownet.pkl")  # Replace with your actual model path
model.eval()

# Load your two images
img1 = cv2.imread(r"C:\Users\pytorch\Desktop\frame_interpolation\vimeo_triplet\sequences\00057\0079\im1.png")
img2 = cv2.imread(r"C:\Users\pytorch\Desktop\frame_interpolation\vimeo_triplet\sequences\00057\0079\im2.png")

# Convert images to tensors and preprocess
img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0

# Specify the number of intermediate frames
n_interpolated_frames = 5

# List to hold all frames, starting with the first image
frames = [img1]

# Generate intermediate frames
for i in range(1, n_interpolated_frames + 1):
    factor = i / (n_interpolated_frames + 1)
    with torch.no_grad():
        interpolated_frame = model.inference(img1_tensor, img2_tensor, factor)
        interpolated_frame = interpolated_frame.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
        interpolated_frame = interpolated_frame.astype(np.uint8)
    frames.append(interpolated_frame)

# Append the final image
frames.append(img2)

# Save frames as individual images
for idx, frame in enumerate(frames):
    cv2.imwrite(f'interpolated_frame_{idx}.jpg', frame)

# Optional: Combine the frames into a video
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (img1.shape[1], img1.shape[0]))

for frame in frames:
    out.write(frame)

out.release()

if __name__ == '__main__':
    img0 = torch.zeros(3, 3, 256, 256).float().to(device)
    img1 = torch.tensor(np.random.normal(0, 1, (3, 3, 256, 256))).float().to(device)
    imgs = torch.cat((img0, img1), 1)

    model = Model()  # This should match the class name you defined for the model.
    model.eval()
    print(model.inference(img0, img1).shape)
