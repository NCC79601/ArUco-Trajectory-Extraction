import os
import cv2
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pickle
from tqdm import tqdm
from utils.trajectory_plotter import plot_trajectory
workdir = os.path.dirname(os.path.dirname(__file__))


def save_frames_to_video(frames: list, output_path: str):
    if not frames:
        print("The frame list is empty.")
        return

    # get the height and width of the frames
    height, width, _ = frames[0].shape

    # define the codec and create a VideoWriter object
    # use avc1 codec for mp4 format
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, 60, (width, height))

    for frame_id, frame in enumerate(frames):
        out.write(frame)

    out.release()

# TODO: rewrite the loading logic, and support specifying different dataset paths
# load dataset.pkl
dataset_path = os.path.join(workdir, 'output', 'dataset.pkl')

with open(dataset_path, 'rb') as file:
    dataset = pickle.load(file)

print(f'len(dataset): {len(dataset)}')

# create directory to save dataset videos
videos_save_dir = os.path.join(workdir, 'output', 'dataset_videos')
if not os.path.exists(videos_save_dir):
    os.makedirs(videos_save_dir)

episode_num = 0

for episode in tqdm(dataset, desc='Saving videos'):
    frames = []
    trajectory = []

    handheld_video_path = episode['handheld_video_path']
    topdown_video_path = episode['topdown_video_path']
    dataframes = episode['dataframes']

    # get frames and trajectory from dataframes
    for dataframe in tqdm(
        dataframes,
        desc='Processing dataframes', 
        leave=False
    ):
        frame = dataframe['frame']
        tvec  = dataframe['tvec']
        rvec  = dataframe['rvec']
        gripper_width = dataframe['gripper_width']

        # resize frame to 1080x1080
        frame = cv2.resize(frame, (1080, 1080))
        cv2.putText(frame, f'width(mm): {gripper_width * 1.0e3:.2f}', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        frames.append(frame)
        trajectory.append({
            'tvec': tvec,
            'rvec': rvec
        })
    
    # save frames as a video
    gripper_video_path = os.path.join(videos_save_dir, f'episode_{episode_num:04d}_gripper.mp4')
    save_frames_to_video(frames, gripper_video_path)

    # plot the trajectory
    source_video_path = topdown_video_path
    trajectory_video_path = os.path.join(videos_save_dir, f'episode_{episode_num:04d}_trajectory.mp4')
    print('Plotting trajectory...')
    plot_trajectory(source_video_path, trajectory, trajectory_video_path, video_height=1080)

    # concatenate two videos
    output_video_path = os.path.join(videos_save_dir, f'episode_{episode_num:04d}_concatenated.mp4')
    os.system(f'ffmpeg -i {gripper_video_path} -i {trajectory_video_path} -filter_complex hstack {output_video_path}')

    episode_num += 1
    

