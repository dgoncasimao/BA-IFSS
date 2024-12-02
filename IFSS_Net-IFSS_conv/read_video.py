import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def split_video_to_frames(video_path, output_folder, flip):
    cap = cv2.VideoCapture(video_path)

    frame_count = 0

    while True:
        ret , frame = cap.read()

        if not ret:
            break

        if flip == True:
            frame = np.fliplr(frame)
        
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.tif")

        cv2.imwrite(frame_filename, frame)

        frame_count += 1
    
    cap.release()

    print(f"Extracted {frame_count} frames from the video")

video_path = "C:/Users/carla/Documents/Master_Thesis/Train_Videos_DLTrack_IFSS/Carla_Videos_DLTrack/TA_Passive/WMA014_TA_Passive1.avi"
output_folder = "C:/Users/carla/Documents/Master_Thesis/Train_Videos_DLTrack_IFSS/Carla_Videos_DLTrack/TA_Passive/WMA014_TA_Passive1/"

split_video_to_frames(video_path, output_folder, flip = False)

