import os
import glob
import yaml
import argparse
from colorama import Fore, Back, Style
try:
    from utils.video_utils import get_start_time, get_duration, crop_video
except ImportError:
    from .utils.video_utils import get_start_time, get_duration, crop_video


def generate_matching(videos_dir: str):
    """
    Generate matching pairs of videos from a directory.
    
    Assume the directory has the following structure:
    - handheld
        - video1.mp4
        - video2.mp4
        - ...
    - topdown
        - video1.mp4
        - video2.mp4
        - ...

    Returns:
        A list of tuples, where each tuple contains two video paths.
    """
    assert os.path.exists(videos_dir), f"Directory {videos_dir} does not exist!"

    handheld_videos_dir = glob.glob(os.path.join(videos_dir, 'handheld/*.mp4')) + \
                          glob.glob(os.path.join(videos_dir, 'handheld/*.MP4'))
    topdown_videos_dir  = glob.glob(os.path.join(videos_dir, 'topdown/*.mp4')) + \
                          glob.glob(os.path.join(videos_dir, 'topdown/*.MP4'))

    try:
        assert len(handheld_videos_dir) == len(topdown_videos_dir)
    except AssertionError:
        print(f"{Fore.YELLOW}[WARN] Number of handheld and topdown videos do not match!{Style.RESET_ALL}")

    handheld_videos = [{
        "path": video,
        "start_time": get_start_time(video)
    } for video in handheld_videos_dir]
    topdown_videos = [{
        "path": video,
        "start_time": get_start_time(video)
    } for video in topdown_videos_dir]
    
    # sort the videos according to start time
    handheld_videos.sort(key=lambda x: x["start_time"])
    topdown_videos.sort( key=lambda x: x["start_time"])

    # generate matching pairs
    matching_pairs = []
    for handheld, topdown in zip(handheld_videos, topdown_videos):
        matching_pairs.append((
            os.path.abspath(handheld["path"]),
            os.path.abspath(topdown["path"])
        ))
    return matching_pairs


def parse_args():
    parser = argparse.ArgumentParser(description='Generate matching pairs of videos')
    parser.add_argument('--videos_dir', type=str, required=True,
                        help='Path to the directory containing handheld and '
                             'topdown videos, must contain two subdirectories: '
                             'handheld and topdown')
    return parser.parse_args()


def main(args):
    matching_pairs = generate_matching(args.videos_dir)
    print(f"Found {len(matching_pairs)} matching pairs of videos")
    output_path = os.path.join(args.videos_dir, "matching_pairs.yaml")
    with open(output_path, "w") as f:
        yaml.dump(matching_pairs, f)


if __name__ == '__main__':
    args = parse_args()
    main(args)

