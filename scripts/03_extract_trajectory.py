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
from concurrent.futures import ThreadPoolExecutor, as_completed

num_workers = multiprocessing.cpu_count()


def extract_trajectory_from_video(video_file: str, aruco_config_file: str):
    '''
    Extracts the trajectory (relative to the first frame) of the camera from the video file. 
    Args:
        video_file: The path to the video file.
        aruco_config_file: The path to the ArUco config file.
    Returns:
        A list of dictionaries, where each dictionary contains the translation vector and rotation vector of the camera related to the first frame.
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
    
    topdown_ids = []
    for tag in aruco_tags:
        if not tag["is_gripper"]:
            topdown_ids.append(tag["id"]) # ids of interest
    id_to_tag_map = {tag['id']: tag for tag in aruco_tags} # id to tag mapping

    # Load the video
    cap = cv2.VideoCapture(video_file)

    # TODO: refactor the file structure
    # load camera calibration
    proj_work_dir = os.path.dirname(os.path.dirname(__file__))
    config_file_path = os.path.join(proj_work_dir, 'configs', 'camera', 'GoPro_11', 'calibration.json')
    with open(config_file_path, 'r') as f:
        camera_config = json.load(f)
    cameraMatrix = np.array(camera_config['K'])
    distCoeffs = np.array(camera_config['D'])

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    trajectory = []

    start_tvec = None
    start_rvec = None
    R_cam_0 = None

    start_frame_id = None
    frame_id = -1

    with tqdm(total=int(total_frames), desc='Extracting', leave=False) as pbar:
        while True:
            # Read a frame from the video
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            pbar.update(1)

            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect ArUco markers
            parameters = aruco.DetectorParameters()
            corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            ids_o_i = []
            if ids is not None:
                for id in ids:
                    if id[0] in topdown_ids:
                        ids_o_i.append(id)

            current_tvec = None
            current_rvec = None

            if len(ids_o_i):
                # Detected tags
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.1, cameraMatrix, distCoeffs) # default marker size 0.1
                
                for i in range(len(ids_o_i)):
                    id  = ids_o_i[i][0]
                    tag = id_to_tag_map[id]

                    marker_size = tag["size"]
                    tvec_cam_tag = tvecs[i][0] * marker_size / 0.1 # convert to real world size
                    rvec_cam_tag = rvecs[i][0]
                    R_cam_tag = cv2.Rodrigues(rvec_cam_tag)[0]

                    # transform to the block frame
                    rvec_block_tag = np.array(tag["rot_vec"], dtype=float)
                    tvec_block_tag = np.array(tag["xyz"], dtype=float)
                    R_block_tag = cv2.Rodrigues(rvec_block_tag)[0]
                    # get the transformation from block to camera
                    tvec_cam_block = tvec_cam_tag - R_cam_tag @ R_block_tag.T @ tvec_block_tag
                    R_cam_block = R_cam_tag @ R_block_tag.T
                    rvec_cam_block = cv2.Rodrigues(R_cam_block)[0].flatten()

                    tvec = tvec_cam_block
                    rvec = rvec_cam_block

                    if len(trajectory) == 0:
                        # Save as the first frame's pose
                        start_frame_id = frame_id
                        start_tvec = tvec
                        start_rvec = rvec
                        R_cam_0    = R_cam_block
                    
                    # Calculate the relative pose
                    R_0_block    = R_cam_0.T @ R_cam_block
                    rvec_0_block = cv2.Rodrigues(R_0_block)[0].flatten()
                    tvec_0_block = R_cam_0.T @ (tvec - start_tvec)

                    if current_tvec is None:
                        current_tvec = tvec_0_block
                        current_rvec = rvec_0_block
                    elif len(trajectory) > 0:
                        # Compare which one is closer to previous frame (in terms of distance)
                        # If closer, then substitute the current pose
                        prev_tvec = trajectory[-1]['tvec']
                        if np.linalg.norm(tvec_0_block - prev_tvec) < np.linalg.norm(current_tvec - prev_tvec):
                            current_tvec = tvec_0_block
                            current_rvec = rvec_0_block
                    else:
                        current_tvec = tvec_0_block
                        current_rvec = rvec_0_block
            
            elif len(trajectory):
                # Lose track of all tags, repeat pose of the last frame
                current_tvec = np.array(trajectory[-1]['tvec'])
                current_rvec = np.array(trajectory[-1]['rvec'])

                print(f'{Fore.YELLOW}[WARN] Frame {frame_id} in video {video_file} lost track of all tags.{Style.RESET_ALL}')
            
            else:
                # If tags are not detected in the first frame, ignore
                print(f'{Fore.YELLOW}[WARN] Frame {frame_id} in video {video_file} lost track of all tags.{Style.RESET_ALL}')
                pass

            # Save the current pose
            if current_tvec is not None and current_rvec is not None:
                current_pose = {
                    'frame_id': frame_id,
                    'tvec':     current_tvec.tolist(),
                    'rvec':     current_rvec.tolist()
                }
                trajectory.append(current_pose)
            elif len(trajectory):
                # If tags are not detected in the current frame, repeat the pose of the last frame
                last_pose = trajectory[-1]
                current_pose = {
                    'frame_id': frame_id,
                    'tvec':     last_pose['tvec'],
                    'rvec':     last_pose['rvec']
                }
                trajectory.append(current_pose)
            else:
                # Skip
                continue
    
    # Release the video capture
    cap.release()

    return {
        'video_path': os.path.abspath(video_file),
        'trajectory': trajectory
    }


def extract_trajectory_from_path(topdown_dir: str, aruco_config_file: str):
    '''
    Extracts trajectories of the camera from the topdown video files.
    Args:
        topdown_dir: The directory containing topdown videos.
        aruco_config_file: The path to the ArUco config file.
    Returns:
        A list of dictionaries, where each dictionary contains the video file path and the trajectory.
    '''
    assert os.path.exists(topdown_dir), f"Directory {topdown_dir} does not exist."
    assert os.path.isdir(topdown_dir), f"{topdown_dir} is not a directory."

    # Get all the video files in the directory
    video_files = glob.glob(os.path.join(topdown_dir, '*.mp4')) + \
                  glob.glob(os.path.join(topdown_dir, '*.MP4'))
    assert len(video_files) > 0, f"No video files found in {topdown_dir}."

    # Extract the trajectory from each video file
    trajectories = []

    with tqdm(total=len(video_files), desc='Processing videos') as pbar:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()

            for video_file in video_files:
                if len(futures) >= num_workers:
                    # limit the number of concurrent processes
                    done, futures = concurrent.futures.wait(
                        futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    
                    # collect the results
                    for future in done:
                        trajectories.append(future.result())
                        pbar.update(1)
                    
                # add new process
                futures.add(executor.submit(
                    extract_trajectory_from_video,
                    video_file,
                    aruco_config_file,
                ))
            
            # wait for the remaining processes to finish
            done, futures = concurrent.futures.wait(futures)
            for future in done:
                trajectories.append(future.result())
                pbar.update(1)
    
    return trajectories


def parse_args():
    parser = argparse.ArgumentParser(description='Extract the trajectories of the camera from topdown videos.')
    parser.add_argument('--topdown_dir', type=str, required=True, help='The directory containing topdown videos.')
    parser.add_argument('--aruco_config_file', type=str, required=True, help='The path to the ArUco config file.')
    parser.add_argument('--trajectory_save_path', type=str, required=True, help='The path to save the extracted trajectoris.')
    return parser.parse_args()


def main(args):
    topdown_dir          = args.topdown_dir
    aruco_config_file    = args.aruco_config_file
    trajectory_save_path = args.trajectory_save_path

    # extract trajectories
    trajectories = extract_trajectory_from_path(topdown_dir, aruco_config_file)

    # save trajectories
    if not os.path.exists(os.path.dirname(trajectory_save_path)):
        os.makedirs(os.path.dirname(trajectory_save_path))
    with open(trajectory_save_path, 'w') as f:
        json.dump(trajectories, f, indent=4)
    print(f"{Fore.GREEN}Trajectories saved at {trajectory_save_path}{Style.RESET_ALL}")


if __name__ == '__main__':
    args = parse_args()
    main(args)