import os
import json
import time
import numpy as np
from utils.vector_plotter import VectorPlotter
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from tqdm import tqdm

with open('trajectory.json', 'r') as f:
    trajectory = json.load(f)

workdir = os.path.dirname(__file__)
save_dir = os.path.join(workdir, 'temp_frames')

save = True
show = False

plotter = VectorPlotter(
    vectors=[np.array([1, 1, 1])],
    origin=[0, 0, 0],
    show=show,
    save=save,
    save_dir=save_dir,
    elev=-125,
    azim=-90
)


left = np.array([1, 0, 0])
up = np.array([0, 1, 0])
forward = np.array([0, 0, 1])

# 暂停标志
is_paused = False

def on_key(event):
    global is_paused
    if event.key == 'p':
        is_paused = not is_paused

# 监听键盘事件
plt.gcf().canvas.mpl_connect('key_press_event', on_key)

for pose in tqdm(trajectory):
    tvec = pose["tvec"]
    rvec = pose["rvec"]
    # print(f' > drawing pose: tvec={tvec}, rvec={rvec}')
    Rot_mat = R.from_rotvec(rvec).as_matrix()
    plotter.update_vectors([
        Rot_mat @ left,
        Rot_mat @ up,
        Rot_mat @ forward
    ], tvec, axis_lim=[-0.4, 0.4], scale=0.3)
    
    # check if paused
    if show:
        while is_paused:
            plt.pause(0.1)

        plt.pause(1.0 / 60)

if save:
    os.system(f'rm -f ./output.mp4')
    os.system(f'rm -f ./concatenated.mp4')

    # concatenate the frames and original videos into a side-by-side video
    os.system(f'ffmpeg -framerate 60 -i {save_dir}/%06d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p -vf "scale=-1:1080" output.mp4')

    os.system(f'ffmpeg -i test_video.mp4 -i output.mp4 -filter_complex "[0:v][1:v]hstack=inputs=2" -c:v libx264 -crf 23 concatenated.mp4')

    # # remove the temporary frames
    # os.system(f'rm -rf {save_dir}')
    # print('Temporary frames removed.')

