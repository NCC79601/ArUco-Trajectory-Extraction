import os
import json
import argparse
import subprocess
from colorama import Fore, Back, Style

workdir = os.path.dirname(__file__)
script_dir = os.path.join(workdir, 'scripts')


def inform_progress(step: str, name: str):
    print(f"{Fore.BLACK}{Back.YELLOW} {step} {Style.RESET_ALL} {Fore.YELLOW}{name}{Style.RESET_ALL}")

def inform_skipped(string: str):
    print(f"{Fore.BLUE}{string}{Style.RESET_ALL}")


def run_pipeline(
    videos_dir: str,
    output_path: str = './output/dataset.pkl',
    temp_dir: str = './temp',
    pipeline_config_path: str = './configs/pipeline_config.json'
):
    # check if the videos directory exists
    assert os.path.exists(videos_dir), f"Videos directory {videos_dir} does not exist."
    
    if os.path.exists(output_path):
        # Ask the user whether to overwrite the file
        print(f"{Fore.YELLOW}[WARN] Output file {output_path} already exists.{Style.RESET_ALL}")
        response = input("Overwrite it? (y/N): ")
        if response.lower() != 'y':
            return
    
    # check if pipeline config file exists
    assert os.path.exists(pipeline_config_path), f"Pipeline config file {pipeline_config_path} does not exist."

    # load pipeline config
    with open(pipeline_config_path, 'r') as f:
        pipeline_config = json.load(f)
    
    downsample_factor = pipeline_config['frame_rate_downsample_factor']
    
    # %% Step 00: Calibrate camera
    inform_progress('Step 00/05', 'Calibrate camera')
    if not pipeline_config['skip_calibrate_camera']:
        calibrate_camera_script = os.path.join(script_dir, '00_calibrate_camera.py')
        subprocess.run([
            'python', calibrate_camera_script,
            '--camera_name', 'GoPro_11',
            '--camera_type', 'pinhole',
        ])
        subprocess.run([
            'python', calibrate_camera_script,
            '--camera_name', 'GoPro_11_fisheye',
            '--camera_type', 'fisheye',
        ])
    else:
        inform_skipped('Skipping calibrate camera...')
    
    # %% Step 01: Generate aruco config
    inform_progress('Step 01/05', 'Generate aruco config')
    if not pipeline_config['skip_generate_aruco_config']:
        generate_aruco_config_script = os.path.join(script_dir, '01_generate_aruco_config.py')
        subprocess.run([
            'python', generate_aruco_config_script,
        ])
    else:
        inform_skipped('Skipping generate aruco config...')

    # %% Step 02: Match videos
    inform_progress('Step 02/05', 'Match videos')
    if not pipeline_config['skip_match_videos']:
        match_videos_script = os.path.join(script_dir, '02_match_videos.py')
        subprocess.run([
            'python', match_videos_script,
            '--videos_dir', videos_dir,
        ])
    else:
        inform_skipped('Skipping match videos...')

    # %% Step 03: Extract trajectory
    inform_progress('Step 03/05', 'Extract trajectory')
    if not pipeline_config['skip_extract_trajectory']:
        extract_trajectory_script = os.path.join(script_dir, '03_extract_trajectory.py')
        subprocess.run([
            'python', extract_trajectory_script,
            '--topdown_dir', os.path.join(videos_dir, 'topdown'),
            '--aruco_config_file', './configs/tag/aruco_config.json',
            '--trajectory_save_path', os.path.join(videos_dir, 'trajectories.json'),
            '--downsample_factor', str(downsample_factor),
        ])
    else:
        inform_skipped('Skipping extract trajectory...')
    
    # %% Step 04: Extract gripper width
    inform_progress('Step 04/05', 'Extract gripper width')
    if not pipeline_config['skip_extract_gripper_width']:
        extract_gripper_width_script = os.path.join(script_dir, '04_extract_gripper_width.py')
        subprocess.run([
            'python', extract_gripper_width_script,
            '--handheld_dir', os.path.join(videos_dir, 'handheld'),
            '--aruco_config_file', './configs/tag/aruco_config.json',
            '--gripper_widths_save_path', os.path.join(videos_dir, 'gripper_widths.json'),
            '--downsample_factor', str(downsample_factor),
        ])
    else:
        inform_skipped('Skipping extract gripper width...')
    
    # %% Step 05: Generate dataset
    inform_progress('Step 05/05', 'Generate dataset')
    if not pipeline_config['skip_generate_dataset']:
        generate_dataset_script = os.path.join(script_dir, '05_generate_dataset.py')
        subprocess.run([
            'python', generate_dataset_script,
            '--matching_pairs_path', os.path.join(videos_dir, 'matching_pairs.json'),
            '--trajectories_path', os.path.join(videos_dir, 'trajectories.json'),
            '--gripper_widths_path', os.path.join(videos_dir, 'gripper_widths.json'),
            '--output_path', output_path,
        ])
    else:
        inform_skipped('Skipping generate dataset...')

# %% main

def parse_arg():
    parser = argparse.ArgumentParser(description='Run the pipeline.')
    parser.add_argument('--videos_dir', type=str, required=True, help='The directory containing the videos.')
    parser.add_argument('--output_path', type=str, default='./output/dataset.pkl', help='The path to save the dataset.')
    parser.add_argument('--temp_dir', type=str, default='./temp', help='The directory to save temporary files.')
    parser.add_argument('--pipeline_config_path', type=str, default='./configs/pipeline_config.json', help='The path to the pipeline config file.')
    return parser.parse_args()


def main(args):
    videos_dir  = args.videos_dir
    output_path = args.output_path
    temp_dir    = args.temp_dir
    pipeline_config_path = args.pipeline_config_path

    run_pipeline(videos_dir, output_path, temp_dir, pipeline_config_path)


if __name__ == '__main__':
    args = parse_arg()
    main(args)
