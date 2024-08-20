import os
import cv2
import cv2.aruco as aruco
import json
import numpy as np
from colorama import Fore, Back, Style
from tqdm import tqdm

# Load the video
video_path = '/Users/max/Desktop/ArUco-Trajectory-Extraction/videos/topdown/GX010409.MP4'
cap = cv2.VideoCapture(video_path)

proj_work_dir = os.path.dirname(os.path.dirname(__file__))
config_file_path = os.path.join(proj_work_dir, 'configs', 'camera', 'GoPro_11', 'calibration.json')
# load camera calibration
with open(config_file_path, 'r') as f:
    camera_config = json.load(f)
cameraMatrix = np.array(camera_config['K'])
distCoeffs = np.array(camera_config['D'])

print(f'{Fore.GREEN}Camera calibration loaded.{Style.RESET_ALL}')

# Define the ArUco dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# Create the video writer
output_path = './test_video.mp4'
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

print(f'{Fore.GREEN}Video writer created, start processing...{Style.RESET_ALL}')

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

with tqdm(total=total_frames, desc='Processing video') as pbar:
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        parameters = aruco.DetectorParameters()
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            # Draw coordinate frames on the markers
            aruco.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, cameraMatrix, distCoeffs)
            for i in range(len(ids)):
                frame = aruco.drawDetectedMarkers(frame, corners, ids)
                frame = cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.05)

        # Write the frame to the output video
        out.write(frame)

        # update progress bar
        pbar.update(1)

# Release the video capture and writer
cap.release()
out.release()