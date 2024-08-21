import os
import json
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R


def save_aruco_config(aruco_config_file: str, aruco_dict: str, aruco_tag_list: list):
    """
    Save ArUco config to json file.
    
    Parameters:
        aruco_config_file: path to the ArUco config file.
        aruco_dict: ArUco dictionary to use.
        aruco_tag_list: list of ArUco tag parameters, including: 
            id:      tag ID
            size:    tag size in meters
            xyz:     tag center position in 3D space (meters)
            rot_vec: tag rotation vector in 3D space
    """
    
    aruco_config = {
        "aruco_dict": aruco_dict,
        "aruco_tags": aruco_tag_list
    }

    if not os.path.exists(os.path.dirname(aruco_config_file)):
        os.makedirs(os.path.dirname(aruco_config_file))
    
    with open(aruco_config_file, 'w') as f:
        json.dump(aruco_config, f, indent=4)
        
    print(f"ArUco config file generated at {aruco_config_file}")



def get_aruco_tag_list():
    """
    Generate ArUco tag list.
    """
    
    aruco_tag_list = []
    
    # tag #08
    R_x_90  = R.from_rotvec(np.array([1., 0., 0.]) * np.pi/2).as_matrix()
    R_z_180 = R.from_rotvec(np.array([0., 0., 1.]) * np.pi).as_matrix()
    R_08_w = R_x_90.T @ R_z_180
    rot_vec_08_w = np.array(R.from_matrix(R_08_w).as_rotvec()).tolist()
    tag_08 = {
        "id": 8,
        "size": 0.06, # meters
        "xyz": [0.0, 0.03, 0.0], # meters
        "rot_vec": rot_vec_08_w,
        "is_gripper": False,
    }
    aruco_tag_list.append(tag_08)

    # tag #09
    R_y_180 = R.from_rotvec(np.array([0., 1., 0.]) * np.pi).as_matrix()
    R_09_w = R_y_180
    rot_vec_09_w = np.array(R.from_matrix(R_09_w).as_rotvec()).tolist()
    tag_09 = {
        "id": 9,
        "size": 0.06, # meters
        "xyz": [0.0, 0.0, -0.03], # meters
        "rot_vec": rot_vec_09_w,
        "is_gripper": False,
    }
    aruco_tag_list.append(tag_09)

    # gripper tags
    # tag #00
    tag_00 = {
        "id": 0,
        "size": 0.02,
        "is_gripper": True
    }
    aruco_tag_list.append(tag_00)

    # tag #01
    tag_01 = {
        "id": 1,
        "size": 0.02,
        "is_gripper": True
    }
    aruco_tag_list.append(tag_01)
        
    return aruco_tag_list


if __name__ == '__main__':
    aruco_tag_list = get_aruco_tag_list()
    workdir = os.path.dirname(os.path.dirname(__file__))
    save_path = os.path.join(workdir, 'configs', 'tag', 'aruco_config.json')
    save_aruco_config(save_path, "DICT_4X4_50", aruco_tag_list)