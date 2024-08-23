import os
import glob
import json
import cv2
from cv2 import aruco
import argparse
import numpy as np
from tqdm import tqdm
from colorama import Fore, Back, Style


def extract_gripper_width_from_video(video_file: str, aruco_config_file: str, generate_undistorted_video=False, output_path='./undistorted_video.mp4', downsample_factor=1):
    '''
    Extracts the gripper width from the video file. 
    Args:
        video_file: The path to the video file.
        aruco_config_file: The path to the ArUco config file.
        generate_undistorted_video: Whether to generate an undistorted video.
        output_path: The path to save the undistorted video.
        downsample_factor: The rate to downsample the video frames. Default is 1.
    Returns:
        A list of gripper widths.
    '''
    assert os.path.exists(video_file), f"Video file {video_file} does not exist."
    assert os.path.exists(aruco_config_file), f"ArUco config file {aruco_config_file} does not exist."

    # Load ArUco config
    with open(aruco_config_file, 'r') as f:
        aruco_config = json.load(f)
    
    aruco_tags = aruco_config['aruco_tags'] # tags
    aruco_dict = aruco_config['aruco_dict']
    if aruco_dict == 'DICT_4X4_50':
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    else:
        raise NotImplementedError(f"ArUco dictionary {aruco_dict} is currently not supported.")
    
    id_of_interest = []
    for tag in aruco_tags:
        if tag["is_gripper"]:
            id_of_interest.append(tag["id"]) # ids of interest (gripper)
    id_to_tag_map = {tag['id']: tag for tag in aruco_tags} # id to tag mapping

    # Load the video
    cap = cv2.VideoCapture(video_file)

    if generate_undistorted_video:
        # Create the video writer
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # TODO: refactor the file structure
    # load camera calibration
    proj_work_dir = os.path.dirname(os.path.dirname(__file__))
    config_file_path = os.path.join(proj_work_dir, 'configs', 'camera', 'GoPro_11_fisheye', 'calibration.json')
    with open(config_file_path, 'r') as f:
        camera_config = json.load(f)
    cameraMatrix = np.array(camera_config['K'])
    distCoeffs = np.array(camera_config['D'])
    # Get pinhole instrinsics after undisortion
    # Adjust 'balance' parameter to change the cropping area
    # Refenrece: https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-part-2-13990f1b157f
    pinhole_cameraMatrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(cameraMatrix, distCoeffs, (1920, 1080), np.eye(3), balance=1.0)
    pinhole_distCoeffs = np.zeros((4, 1))

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(cameraMatrix, distCoeffs, np.eye(3), pinhole_cameraMatrix, camera_config['DIM'], cv2.CV_16SC2)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    gripper_width_list = []

    frame_id = 0

    with tqdm(total=int(total_frames / downsample_factor), desc='Extracting', leave=False) as pbar:
        while True:
            # Read a frame from the video
            for i in range(downsample_factor):
                ret, frame = cap.read()
                if not ret:
                    break
            if not ret:
                break
            # Undistort fisheye image to pinhole image
            # frame = cv2.fisheye.undistortImage(frame, cameraMatrix, distCoeffs)
            frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect ArUco markers
            parameters = aruco.DetectorParameters()
            corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            gripper_tag_pos = {
                "frame_id": frame_id,
                "left_gripper_pos": None,
                "right_gripper_pos": None
            }

            frame_id += 1

            if ids is not None:
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.1, pinhole_cameraMatrix, pinhole_distCoeffs) # default marker size 0.1
                
                for i in range(len(ids)):
                    if ids[i][0] not in id_of_interest:
                        continue

                    id  = ids[i][0]
                    tag = id_to_tag_map[id]
                    marker_size = tag["size"]
                    tvec_cam_tag = tvecs[i][0] * marker_size / 0.1 # convert to real world size
                    
                    if id == 0:
                        gripper_tag_pos["left_gripper_pos"] = tvec_cam_tag.tolist()
                    elif id == 1:
                        gripper_tag_pos["right_gripper_pos"] = tvec_cam_tag.tolist()
                    
                    frame = cv2.drawFrameAxes(frame, pinhole_cameraMatrix, pinhole_distCoeffs, rvecs[i], tvecs[i], 0.05)
            
            if generate_undistorted_video:
                out.write(frame)

            if gripper_tag_pos["left_gripper_pos"] is not None and gripper_tag_pos["right_gripper_pos"] is not None:
                gripper_width_list.append({
                    "frame_id": gripper_tag_pos["frame_id"],
                    "gripper_width": gripper_tag_pos["right_gripper_pos"][0] - gripper_tag_pos["left_gripper_pos"][0]
                })
            elif len(gripper_width_list) > 0:
                # If one of the gripper tags is not detected, use the previous frame's position
                gripper_width_list.append(gripper_width_list[-1])
            else:
                # Skip
                pass
            
            pbar.update(1)
    
    # Release the video capture
    cap.release()

    if generate_undistorted_video:
        out.release()

    return gripper_width_list


def extract_gripper_width_from_path(handheld_dir: str, aruco_config_file: str, downsample_factor: int = 1):
    '''
    Extracts gripper widths from all the videos in the directory.
    Args:
        handheld_dir: The directory containing handheld videos.
        aruco_config_file: The path to the ArUco config file.
        downsample_factor: The rate to downsample the video frames. Default is 1.
    Returns:
        A list of dictionaries, where each dictionary contains the video file path and the gripper widths.
    '''
    assert os.path.exists(handheld_dir), f"Directory {handheld_dir} does not exist."
    assert os.path.isdir(handheld_dir), f"{handheld_dir} is not a directory."

    # Get all the video files in the directory
    video_files = glob.glob(os.path.join(handheld_dir, '*.mp4')) + \
                  glob.glob(os.path.join(handheld_dir, '*.MP4'))
    assert len(video_files) > 0, f"No video files found in {handheld_dir}."

    # Extract the trajectory from each video file
    gripper_widths = []
    for video_file in tqdm(video_files, desc='Processing videos'):
        gripper_width = extract_gripper_width_from_video(video_file, aruco_config_file, downsample_factor=downsample_factor)
        gripper_widths.append({
            'video_path': os.path.abspath(video_file),
            'gripper_width': gripper_width
        })
    
    return gripper_widths


def parse_args():
    parser = argparse.ArgumentParser(description='Extract gripper widths from handheld videos.')
    parser.add_argument('--handheld_dir', type=str, required=True, help='The directory containing handheld videos.')
    parser.add_argument('--aruco_config_file', type=str, required=True, help='The path to the ArUco config file.')
    parser.add_argument('--gripper_widths_save_path', type=str, default='./gripper_widths.json', help='The path to save gripper widths.')
    parser.add_argument('--downsample_factor', type=int, default=1, help='The rate at which to downsample the video frames.')
    return parser.parse_args()


def main(args):
    handheld_dir             = args.handheld_dir
    aruco_config_file        = args.aruco_config_file
    gripper_widths_save_path = args.gripper_widths_save_path
    downsample_factor        = args.downsample_factor

    # extract trajectories
    trajectories = extract_gripper_width_from_path(handheld_dir, aruco_config_file, downsample_factor)

    # save trajectories
    if not os.path.exists(os.path.dirname(gripper_widths_save_path)):
        os.makedirs(os.path.dirname(gripper_widths_save_path))
    with open(gripper_widths_save_path, 'w') as f:
        json.dump(trajectories, f, indent=4)
    print(f"{Fore.GREEN}Trajectories saved at {gripper_widths_save_path}{Style.RESET_ALL}")


if __name__ == '__main__':
    args = parse_args()
    main(args)