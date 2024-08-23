import os
import json
import argparse
from colorama import Fore, Back, Style
import cv2
import pickle
from tqdm import tqdm


def generate_dataset(
    matching_pairs_path: str,
    trajectories_path:   str,
    gripper_widths_path: str,
):
    assert os.path.exists(matching_pairs_path), f"Matching pairs file {matching_pairs_path} does not exist."
    assert os.path.exists(trajectories_path), f"Trajectories file {trajectories_path} does not exist."
    assert os.path.exists(gripper_widths_path), f"Gripper widths file {gripper_widths_path} does not exist."


    with open(matching_pairs_path, 'r') as f:
        matching_pairs = json.load(f)
    
    with open(trajectories_path, 'r') as f:
        trajectories = json.load(f)

    with open(gripper_widths_path, 'r') as f:
        gripper_widths = json.load(f)

    dataset = []

    for pairs in tqdm(matching_pairs, desc='Generating dataset'):
        handheld_path = pairs['handheld']
        topdown_path  = pairs['topdown']

        # find the trajectory for the handheld video
        found = False
        for trajectory in trajectories:
            if trajectory['video_path'] == topdown_path:
                trajectory = trajectory['trajectory']
                found = True
                break
        
        if not found:
            print(f'{Fore.YELLOW}[WARN] Trajectory for {topdown_path} not found.{Style.RESET_ALL}')
            continue
        
        # find the gripper width for the handheld video
        found = False
        for gripper_width in gripper_widths:
            if gripper_width['video_path'] == handheld_path:
                gripper_width = gripper_width['gripper_width']
                found = True
                break
        
        if not found:
            print(f'{Fore.YELLOW}[WARN] Gripper width for {handheld_path} not found.{Style.RESET_ALL}')
            continue

        current_episode = []

        i = 0 # for topdown video
        j = 0 # for handheld video
        
        cap = cv2.VideoCapture(handheld_path)
        # read first frame
        ret, frame = cap.read()
        
        with tqdm(
            total=min(len(trajectory), len(gripper_width)),
            desc='Processing frames',
            leave=False
        ) as pbar:
            while i < len(trajectory) and j < len(gripper_width):
                traj_frame_id = trajectory[i]['frame_id']
                grip_frame_id = gripper_width[j]['frame_id']

                if traj_frame_id != grip_frame_id:
                    if traj_frame_id < grip_frame_id:
                        i += 1
                    else:
                        j += 1
                    continue
                
                # TODO: downsample the frame to 224x224
                frame = cv2.resize(frame, (224, 224))

                current_episode.append({
                    'frame': frame,
                    'tvec': trajectory[i]['tvec'],
                    'rvec': trajectory[i]['rvec'],
                    'gripper_width': gripper_width[j]['gripper_width']
                })
                
                i += 1
                j += 1
                
                # read next frame
                ret, frame = cap.read()

                pbar.update(1)

            cap.release()   
            dataset.append({
                'handheld_video_path': handheld_path,
                'topdown_video_path':  topdown_path,
                'dataframes': current_episode
            })

    return dataset


def arg_parse():
    parser = argparse.ArgumentParser(description='Generate the dataset.')
    parser.add_argument('--matching_pairs_path', type=str, required=True, help='The path to the matching pairs file.')
    parser.add_argument('--trajectories_path', type=str, required=True, help='The path to the trajectories file.')
    parser.add_argument('--gripper_widths_path', type=str, required=True, help='The path to the gripper widths file.')
    parser.add_argument('--output_path', type=str, default='./dataset.pkl', help='The path to save the dataset.')
    return parser.parse_args()


def main(args):
    matching_pairs_path = args.matching_pairs_path
    trajectories_path   = args.trajectories_path
    gripper_widths_path = args.gripper_widths_path
    output_path         = args.output_path

    dataset = generate_dataset(matching_pairs_path, trajectories_path, gripper_widths_path)

    # Save dataset as pickle file
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)


if __name__ == '__main__':
    args = arg_parse()
    main(args)