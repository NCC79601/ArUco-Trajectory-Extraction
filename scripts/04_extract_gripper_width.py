import concurrent.futures
import os
import glob
import json
import cv2
from cv2 import aruco
import argparse
import multiprocessing
import numpy as np
from tqdm import tqdm
from colorama import Fore, Back, Style
import concurrent
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.logger import get_logger

num_workers = multiprocessing.cpu_count()


def extract_gripper_width_from_video(video_file: str, aruco_config_file: str, generate_undistorted_video=False, output_path='./undistorted_video.mp4', logger=None):
    '''
    Extracts the gripper width from the video file. 
    Args:
        video_file: The path to the video file.
        aruco_config_file: The path to the ArUco config file.
        generate_undistorted_video: Whether to generate an undistorted video.
        output_path: The path to save the undistorted video.
        logger: The logger object.
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
    
    gripper_ids = []
    for tag in aruco_tags:
        if tag["is_gripper"]:
            gripper_ids.append(tag["id"]) # ids of interest (gripper)
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

    frame_id = -1

    with tqdm(total=int(total_frames), desc='Extracting', leave=False) as pbar:
        while True:
            # Read a frame from the video
            ret, frame = cap.read()
            frame_id += 1
            if not ret:
                break
            # Undistort fisheye image to pinhole image
            frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect ArUco markers
            parameters = aruco.DetectorParameters()
            corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            ids_o_i = []
            if ids is not None:
                for id in ids:
                    if id[0] in gripper_ids:
                        ids_o_i.append(id)

            gripper_tag_pos = {
                "frame_id": frame_id,
                "left_gripper_pos": None,
                "right_gripper_pos": None
            }

            if len(ids_o_i):
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.1, pinhole_cameraMatrix, pinhole_distCoeffs) # default marker size 0.1
                
                for i in range(len(ids_o_i)):
                    if ids_o_i[i][0] not in gripper_ids:
                        continue

                    id  = ids_o_i[i][0]
                    tag = id_to_tag_map[id]
                    marker_size = tag["size"]
                    tvec_cam_tag = tvecs[i][0] * marker_size / 0.1 # convert to real world size
                    
                    if id == 0:
                        gripper_tag_pos["left_gripper_pos"] = tvec_cam_tag.tolist()
                    elif id == 1:
                        gripper_tag_pos["right_gripper_pos"] = tvec_cam_tag.tolist()
                    
                    if generate_undistorted_video:
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
                last = gripper_width_list[-1]
                gripper_width_list.append({
                    "frame_id": gripper_tag_pos["frame_id"],
                    "gripper_width": last["gripper_width"]
                })

                print(f'{Fore.YELLOW}[WARN] Frame {frame_id} in video {video_file} lost track of all tags.{Style.RESET_ALL}')
                if logger is not None:
                    logger.warning(f'Frame {frame_id} in video {video_file} lost track of all tags.')
            else:
                # Skip
                print(f'{Fore.YELLOW}[WARN] Frame {frame_id} in video {video_file} lost track of all tags.{Style.RESET_ALL}')
                if logger is not None:
                    logger.warning(f'Frame {frame_id} in video {video_file} lost track of all tags.')
            
            pbar.update(1)
    
    # Release the video capture
    cap.release()

    if generate_undistorted_video:
        out.release()

    return {
        'video_path': os.path.abspath(video_file),
        'gripper_width': gripper_width_list
    }


def extract_gripper_width_from_path(handheld_dir: str, aruco_config_file: str, logger=None):
    '''
    Extracts gripper widths from all the videos in the directory.
    Args:
        handheld_dir: The directory containing handheld videos.
        aruco_config_file: The path to the ArUco config file.
        logger: The logger object.
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

    with tqdm(total=len(video_files), desc='Processing videos') as pbar:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = set()

            for video_file in video_files:
                if len(futures) >= num_workers:
                    done, futures = concurrent.futures.wait(
                        futures,
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )

                    for future in done:
                        gripper_widths.append(future.result())
                        pbar.update(1)
                
                # add new process
                futures.add(executor.submit(
                    extract_gripper_width_from_video,
                    video_file,
                    aruco_config_file,
                    logger
                ))
            
            # wait for the remaining process to finish
            done, futures = concurrent.futures.wait(futures)

            for future in done:
                gripper_widths.append(future.result())
                pbar.update(1)

    return gripper_widths


def parse_args():
    parser = argparse.ArgumentParser(description='Extract gripper widths from handheld videos.')
    parser.add_argument('--handheld_dir', type=str, required=True, help='The directory containing handheld videos.')
    parser.add_argument('--aruco_config_file', type=str, required=True, help='The path to the ArUco config file.')
    parser.add_argument('--gripper_widths_save_path', type=str, default='./gripper_widths.json', help='The path to save gripper widths.')
    parser.add_argument('--logger_name', type=str, default='default_logger', help='The name of the logger.')
    return parser.parse_args()


def main(args):
    handheld_dir             = args.handheld_dir
    aruco_config_file        = args.aruco_config_file
    gripper_widths_save_path = args.gripper_widths_save_path
    logger_name              = args.logger_name

    logger = get_logger(logger_name)

    # extract trajectories
    trajectories = extract_gripper_width_from_path(handheld_dir, aruco_config_file, logger)

    # save trajectories
    if not os.path.exists(os.path.dirname(gripper_widths_save_path)):
        os.makedirs(os.path.dirname(gripper_widths_save_path))
    with open(gripper_widths_save_path, 'w') as f:
        json.dump(trajectories, f, indent=4)
    print(f"{Fore.GREEN}Trajectories saved at {gripper_widths_save_path}{Style.RESET_ALL}")


if __name__ == '__main__':
    args = parse_args()
    main(args)