import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.trajectory_plotter import plot_trajectory
import json
import argparse

workdir = os.path.dirname(os.path.dirname(__file__))
script_path = os.path.join(workdir, 'utils', 'plot_trajectory.py')


def get_filename_without_extension(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def plot(
    trajectories_file_path: str,
    source_video_path: str,
    output_path: str
):
    assert os.path.exists(trajectories_file_path), f"Trajectories file {trajectories_file_path} does not exist."
    assert os.path.exists(source_video_path), f"Source video {source_video_path} does not exist."
    
    with open(trajectories_file_path, 'r') as f:
        trajectories = json.load(f)

    source_video_filename = get_filename_without_extension(source_video_path)

    found = False

    for episode in trajectories:
        video_file = episode['video_file']
        trajectory = episode['trajectory']

        if get_filename_without_extension(video_file) == source_video_filename:
            # found the target video
            found = True
            plot_trajectory(source_video_path, trajectory, output_path)
    
    if not found:
        raise FileNotFoundError(f"Video file {source_video_filename} not found in trajectories file.")


def arg_parse():
    parser = argparse.ArgumentParser(description='Plot the trajectory of the camera.')
    parser.add_argument('--trajectories_file_path', type=str, required=True, help='The path to the trajectories file.')
    parser.add_argument('--source_video_path', type=str, required=True, help='The path to the source video.')
    parser.add_argument('--output_path', type=str, default='plot_trajectory_output.mp4', help='The path to save the output video.')
    return parser.parse_args()


def main(args):
    trajectories_file_path = args.trajectories_file_path
    source_video_path = args.source_video_path
    output_path = args.output_path

    plot(trajectories_file_path, source_video_path, output_path)


if __name__ == '__main__':
    args = arg_parse()
    main(args)