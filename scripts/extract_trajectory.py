import os
import glob
import json
import cv2
from cv2 import aruco
import numpy as np
from tqdm import tqdm


def extract_trajectory_from_video(video_file: str, aruco_config_file: str):
    '''
    Extracts the trajectory (relative to the first frame) of the camera from the video file. 
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
    id_of_interest = [tag['id'] for tag in aruco_tags] # ids of interest
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

    first_tvec = None
    first_rvec = None
    R_cam_0 = None

    is_first_frame = True

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

            current_tvec = None
            current_rvec = None

            if ids is not None:
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, cameraMatrix, distCoeffs)
                
                for i in range(len(ids)):
                    if ids[i][0] not in id_of_interest:
                        continue

                    id = ids[i][0]
                    tvec_cam_tag = tvecs[i][0]
                    rvec_cam_tag = rvecs[i][0]
                    R_cam_tag = cv2.Rodrigues(rvec_cam_tag)[0]

                    tag = id_to_tag_map[id]

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

                    if is_first_frame:
                        # Save as the first frame's pose
                        first_tvec = tvec
                        first_rvec = rvec
                        R_cam_0    = R_cam_block
                        break
                    
                    # Calculate the relative pose
                    R_0_block    = R_cam_0.T @ R_cam_block
                    rvec_0_block = cv2.Rodrigues(R_0_block)[0].flatten()
                    tvec_0_block = R_cam_0.T @ (tvec - first_tvec)

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
                current_tvec = trajectory[-1]['tvec']
                current_rvec = trajectory[-1]['rvec']
            
            else:
                # If tags are not detected in the first frame, skip
                continue

            if is_first_frame:
                is_first_frame = False
            else:
                # Save the current pose
                current_pose = {
                    'tvec': current_tvec.tolist(),
                    'rvec': current_rvec.tolist()
                }
                trajectory.append(current_pose)
            
            pbar.update(1)
    
    # Release the video capture
    cap.release()

    return trajectory


def extract_trajectory_from_path(topdown_path: str):
    pass



def parse_args():
    pass


def main(args):
    pass


def test():
    trajectory = extract_trajectory_from_video(
        '/Users/max/Desktop/ArUco-Trajectory-Extraction/videos/topdown/GX010409.MP4', \
        '/Users/max/Desktop/ArUco-Trajectory-Extraction/configs/tag/aruco_config.json'
    )
    return trajectory


if __name__ == '__main__':
    trajectory = test()
    with open('trajectory.json', 'w') as f:
        json.dump(trajectory, f, indent=4)

# if __name__ == '__main__':
#     args = parse_args()
#     main(args)