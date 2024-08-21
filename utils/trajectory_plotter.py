import os
import sys
import json
import argparse
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.vector_plotter import VectorPlotter
from scipy.spatial.transform import Rotation as R
from colorama import Fore, Back, Style
from tqdm import tqdm


def plot_trajectory(
    source_video_path: str,
    trajectory: list,
    output_path: str,
    temp_dir: str = './temp',
    video_height: int = 1080,
    elev: int = -125, 
    azim: int = -90
):
    assert os.path.exists(source_video_path), f"Source video {source_video_path} does not exist."
    
    if os.path.exists(temp_dir):
        raise FileExistsError(f"Temporary directory {temp_dir} already exists!")
    
    os.makedirs(temp_dir)
    
    temp_frame_dir = os.path.join(temp_dir, 'frames')
    temp_video_dir = os.path.join(temp_dir, 'source_video')
    if not os.path.exists(temp_frame_dir):
        os.makedirs(temp_frame_dir)
    if not os.path.exists(temp_video_dir):
        os.makedirs(temp_video_dir)
    
    plot_video_path = os.path.join(temp_video_dir, 'plot_video.mp4')
    converted_video_path = os.path.join(temp_video_dir, 'source_video.mp4')

    plotter = VectorPlotter(
        vectors=[np.array([1, 1, 1])],
        origin=[0, 0, 0],
        show=False,
        save=True,
        save_dir=temp_frame_dir,
        elev=elev,
        azim=azim
    )

    # construct coordinate frame
    left    = np.array([1, 0, 0])
    up      = np.array([0, 1, 0])
    forward = np.array([0, 0, 1])

    for pose in tqdm(trajectory):
        tvec = pose["tvec"]
        rvec = pose["rvec"]
        Rot_mat = R.from_rotvec(rvec).as_matrix()
        plotter.update_vectors([
            Rot_mat @ left,
            Rot_mat @ up,
            Rot_mat @ forward
        ], tvec, axis_lim=[-0.4, 0.4], scale=0.3)

    # Use ffmpeg to convert the plotted frames to a video
    os.system(f'ffmpeg -framerate 60 -i {temp_frame_dir}/%06d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p -vf "scale=-1:{video_height}" {plot_video_path}')

    # Use ffmpeg to convert the source video to the specified resolution
    os.system(f'ffmpeg -i {source_video_path} -vf "scale=-1:{video_height}" {converted_video_path}')

    # Concatenate the frames and source videos into a side-by-side video
    os.system(f'ffmpeg -i {converted_video_path} -i {plot_video_path} -filter_complex "[0:v][1:v]hstack=inputs=2" -c:v libx264 -crf 23 {output_path}')

    # Remove temporary files
    os.system(f'rm -r {temp_dir}')
    print(f'{Fore.BLUE}Plot video saved to {output_path}{Style.RESET_ALL}')