import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.callibrator import Callibrator


def parse_args():
    parser = argparse.ArgumentParser(description='Calibrate camera')
    parser.add_argument('--camera_name', type=str, default='GoPro_11',
                        help='Name of the camera used')
    parser.add_argument('--camera_type', type=str, default='pinhole',
                        help='Type of the camera used, pinhole or fisheye')
    return parser.parse_args()


def main(args):
    # specify workdir
    root_dir = os.path.dirname(os.path.dirname(__file__))
    camera_workdir = os.path.join(root_dir, 'configs', 'camera', args.camera_name)
    
    # perform callibration
    callibrator = Callibrator(camera_name=args.camera_name, camera_type=args.camera_type)
    callibrator.calibrate(images=os.path.join(camera_workdir, 'images'))
    callibrator.save_calibration(os.path.join(camera_workdir, 'calibration.json'))


if __name__ == '__main__':
    args = parse_args()
    main(args)