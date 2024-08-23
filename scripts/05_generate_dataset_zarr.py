import os
import json
import argparse
from colorama import Fore, Back, Style
import cv2
import av
import concurrent.futures
import multiprocessing
import zarr
import pickle
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.from_umi.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
register_codecs()
from utils.from_umi.replay_buffer import ReplayBuffer


# parameters used in UMI's pipeline
compression_level = 99
out_res = (224, 224)
mirror_swap = False
num_workers = multiprocessing.cpu_count()
fisheye_converter = None


def generate_dataset(
    matching_pairs_path: str,
    trajectories_path:   str,
    gripper_widths_path: str,
    output_path:         str = './dataset.zarr'
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

    out_replay_buffer = ReplayBuffer.create_empty_zarr(
        storage=zarr.MemoryStore())
    videos_dict = defaultdict(list)
    all_videos = set()
    vid_args = list()
    buffer_start = 0

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

        episode_data = dict()

        eef_pos = []
        eef_rot = []
        _gripper_widths = []
        video_start = 0
        video_end = 0

        i = 0 # for topdown video
        j = 0 # for handheld video
        
        # cap = cv2.VideoCapture(handheld_path)
        # read first frame
        # ret, frame = cap.read()
        
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
                        video_start =j
                    continue
                
                # TODO: downsample the frame to 224x224
                # frame = cv2.resize(frame, (224, 224))

                eef_pos.append(trajectory[i]['tvec'])
                eef_rot.append(trajectory[i]['rvec'])
                _gripper_widths.append(gripper_width[j]['gripper_width'])
                
                i += 1
                j += 1
                video_end = j
                
                # read next frame
                # ret, frame = cap.read()

                pbar.update(1)

            # cap.release()
                           
            demo_start_pose = np.empty((len(eef_pos), 6))
            demo_end_pose   = np.empty_like(demo_start_pose)

            demo_start_pose[:] = np.array([0, 0, 0, 0, 0, 0])
            demo_end_pose[:] = np.concatenate([eef_pos[-1], eef_rot[-1]])

            episode_data['robot0_eef_pos'] = np.array(eef_pos)
            episode_data['robot0_eef_rot'] = np.array(eef_rot)
            episode_data['robot0_gripper_width'] = np.array(_gripper_widths)
            episode_data['robot0_demo_start_pose'] = demo_start_pose
            episode_data['robot0_demo_end_pose'] = demo_end_pose

            print(f'episode data: {episode_data}')
            print(f' > component shapes:')
            for k, v in episode_data.items():
                print(f'   - {k}: {v.shape}')

            print(f'adding to replay buffer...')
            out_replay_buffer.add_episode(data=episode_data, compressors=None)
            print(f'successfully added to replay buffer!')

            videos_dict[str(handheld_path)].append({
                'camera_idx': 0,
                'frame_start': video_start,
                'frame_end': video_end,
                'buffer_start': buffer_start
            })
            # print(f'videos_dict: {videos_dict}') 
            buffer_start += video_end - video_start
        
        vid_args.extend(videos_dict.items())
        all_videos.update(videos_dict.keys())
    
    print(f"[UMI] {len(all_videos)} videos used in total!")

    # get image size
    with av.open(vid_args[0][0]) as container:
        in_stream = container.streams.video[0]
        ih, iw = in_stream.height, in_stream.width
    
    # dump images
    img_compressor = JpegXl(level=compression_level, numthreads=1)
    cam_id = 0
    name = f'camera{cam_id}_rgb'
    _ = out_replay_buffer.data.require_dataset(
        name=name,
        shape=(out_replay_buffer['robot0_eef_pos'].shape[0],) + out_res + (3,),
        chunks=(1,) + out_res + (3,),
        compressor=img_compressor,
        dtype=np.uint8
    )

    def get_image_transform(in_res, out_res, crop_ratio:float = 1.0, bgr_to_rgb: bool=False):
        iw, ih = in_res
        ow, oh = out_res
        ch = round(ih * crop_ratio)
        cw = round(ih * crop_ratio / oh * ow)
        interp_method = cv2.INTER_AREA

        w_slice_start = (iw - cw) // 2
        w_slice = slice(w_slice_start, w_slice_start + cw)
        h_slice_start = (ih - ch) // 2
        h_slice = slice(h_slice_start, h_slice_start + ch)
        c_slice = slice(None)
        if bgr_to_rgb:
            c_slice = slice(None, None, -1)

        def transform(img: np.ndarray):
            assert img.shape == ((ih,iw,3))
            # crop
            img = img[h_slice, w_slice, c_slice]
            # resize
            img = cv2.resize(img, out_res, interpolation=interp_method)
            return img
        
        return transform

    def video_to_zarr(replay_buffer, mp4_path, tasks):
        # pkl_path = os.path.join(os.path.dirname(mp4_path), 'tag_detection.pkl')
        # tag_detection_results = pickle.load(open(pkl_path, 'rb'))
        resize_tf = get_image_transform(
            in_res=(iw, ih),
            out_res=out_res
        )
        tasks = sorted(tasks, key=lambda x: x['frame_start'])
        camera_idx = None
        for task in tasks:
            if camera_idx is None:
                camera_idx = task['camera_idx']
            else:
                assert camera_idx == task['camera_idx']
        name = f'camera{camera_idx}_rgb'
        img_array = replay_buffer.data[name]
        
        curr_task_idx = 0
        
        is_mirror = None
        if mirror_swap:
            ow, oh = out_res
            mirror_mask = np.ones((oh,ow,3),dtype=np.uint8)
            # mirror_mask = draw_predefined_mask(
            #     mirror_mask, color=(0,0,0), mirror=True, gripper=False, finger=False)
            is_mirror = (mirror_mask[...,0] == 0)
        
        with av.open(mp4_path) as container:
            in_stream = container.streams.video[0]
            # in_stream.thread_type = "AUTO"
            in_stream.thread_count = 1
            buffer_idx = 0
            for frame_idx, frame in tqdm(enumerate(container.decode(in_stream)), total=in_stream.frames, leave=False):
                if curr_task_idx >= len(tasks):
                    # all tasks done
                    break
                
                if frame_idx < tasks[curr_task_idx]['frame_start']:
                    # current task not started
                    continue
                elif frame_idx < tasks[curr_task_idx]['frame_end']:
                    if frame_idx == tasks[curr_task_idx]['frame_start']:
                        buffer_idx = tasks[curr_task_idx]['buffer_start']
                    
                    # do current task
                    img = frame.to_ndarray(format='rgb24')

                    # inpaint tags
                    # this_det = tag_detection_results[frame_idx]
                    # all_corners = [x['corners'] for x in this_det['tag_dict'].values()]
                    # for corners in all_corners:
                    #     img = inpaint_tag(img, corners)
                        
                    # mask out gripper
                    # img = draw_predefined_mask(img, color=(0,0,0), 
                    #     mirror=no_mirror, gripper=True, finger=False)
                    # resize
                    if fisheye_converter is None:
                        img = resize_tf(img)
                    else:
                        img = fisheye_converter.forward(img)
                        
                    # handle mirror swap
                    if mirror_swap:
                        img[is_mirror] = img[:,::-1,:][is_mirror]
                        
                    # compress image
                    img_array[buffer_idx] = img
                    buffer_idx += 1
                    
                    if (frame_idx + 1) == tasks[curr_task_idx]['frame_end']:
                        # current task done, advance
                        curr_task_idx += 1
                else:
                    assert False

    # compress videos to zarr 
    with tqdm(total=len(vid_args)) as pbar:
        # one chunk per thread, therefore no synchronization needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            for mp4_path, tasks in vid_args:
                if len(futures) >= num_workers:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    pbar.update(len(completed))

                futures.add(executor.submit(video_to_zarr, 
                    out_replay_buffer, mp4_path, tasks))

            completed, futures = concurrent.futures.wait(futures)
            pbar.update(len(completed))

    print([x.result() for x in completed])

    # dump to disk
    output = output_path
    print(f"Saving ReplayBuffer to {output}")
    with zarr.ZipStore(output, mode='w') as zip_store:
        out_replay_buffer.save_to_store(
            store=zip_store
        )
    print(f"Done! {len(all_videos)} videos used in total!")


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

    generate_dataset(matching_pairs_path, trajectories_path, gripper_widths_path, output_path)


if __name__ == '__main__':
    args = arg_parse()
    main(args)