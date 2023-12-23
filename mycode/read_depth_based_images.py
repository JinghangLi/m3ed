import os
import h5py
import argparse

from PIL.ImageDraw import ImageDraw
from tqdm import tqdm
from PIL import Image
import yaml
import pandas as pd
import numpy as np
import cv2 as cv
import open3d as o3d


def get_strttc_data(args):
    save_path = f'/home/jhang/workspace/STRTTC_RAL2023/code/STRTTC_matlab/dataset/real/M3ED/{args.seq_name}'
    root = save_path
    name_i = 1
    while os.path.exists(root):
        root = f"{save_path}_{name_i}"
        name_i += 1

    e_save = f"{root}/txt"
    rgb_save = f"{root}/rgb/frames"
    e_image_save = f"{root}/txt_image"

    os.makedirs(e_save, exist_ok=True)
    os.makedirs(rgb_save, exist_ok=True)
    os.makedirs(e_image_save, exist_ok=True)

    with h5py.File(args.data_path, 'r') as h5f:
        with open(f"{rgb_save}/timestamps.txt", 'w') as f:
            ts_start = h5f['/ovc/ts'][args.events_range[0]]
            ts_end = h5f['/ovc/ts'][args.events_range[1]]
            for idx in tqdm(range(args.events_range[0], args.events_range[1])):
                name_id = "{:0>10d}".format(int(idx))
                gray_left = h5f['/ovc/left/data'][idx]
                I_left_pil = Image.fromarray(gray_left.squeeze())
                I_left_pil.save(f"{rgb_save}/frame_{name_id}.png")

                if idx == args.events_range[1]-1:
                    I_left_pil.save(f"/mnt/disk1/M3ED/sample/{args.seq_name}_{name_id}.png")


                e_idx_start = h5f['/ovc/ts_map_prophesee_left_t'][idx - 1]
                e_idx_end = h5f['/ovc/ts_map_prophesee_left_t'][idx]

                left_events_x = h5f['/prophesee/left/x'][e_idx_start:e_idx_end]
                left_events_y = h5f['/prophesee/left/y'][e_idx_start:e_idx_end]
                left_events_t = h5f['/prophesee/left/t'][e_idx_start:e_idx_end]
                left_events_p = h5f['/prophesee/left/p'][e_idx_start:e_idx_end]
                # Event visualization
                events_image = np.ones(
                    (h5f['/prophesee/left/calib/resolution'][1], h5f['/prophesee/left/calib/resolution'][0], 3),
                    dtype=np.dtype("uint8"))
                events_image *= 255
                for i in range(len(left_events_x)):
                    # [t, x, y, pol] events array
                    if left_events_p[i] == 1:
                        events_image[int(left_events_y[i]), int(left_events_x[i]), :] = [0, 0, 255]  # Blue
                    else:
                        events_image[int(left_events_y[i]), int(left_events_x[i]), :] = [255, 0, 0]  # Red
                cv.imwrite(f"{e_image_save}/{name_id}.png", cv.cvtColor(events_image, cv.COLOR_RGB2BGR))

                e_array_df = pd.DataFrame(
                    {'t': left_events_t, 'x': left_events_x, 'y': left_events_y, 'p': left_events_p})
                e_array_df.to_csv(f"{e_save}/events_{name_id}.txt", sep=' ', index=False, header=False)
                del left_events_x, left_events_y, left_events_t, left_events_p

                ts = h5f['/ovc/ts'][idx]
                f.write(f'{idx:10} {ts}\n')

        calib_data = {
            'ovc': {
                'left': {
                    'resolution': h5f['/ovc/left/calib/resolution'][:].tolist(),
                    'T_to_prophesee_left': h5f['/ovc/left/calib/T_to_prophesee_left'][:].tolist(),
                    'instrinsics': h5f['/ovc/left/calib/intrinsics'][:].tolist(),
                    'distortion_coeffs': h5f['/ovc/left/calib/distortion_coeffs'][:].tolist()
                },
                'right': {
                    'resolution': h5f['/ovc/right/calib/resolution'][:].tolist(),
                    'T_to_prophesee_left': h5f['/ovc/right/calib/T_to_prophesee_left'][:].tolist(),
                    'instrinsics': h5f['/ovc/right/calib/intrinsics'][:].tolist(),
                    'distortion_coeffs': h5f['/ovc/right/calib/distortion_coeffs'][:].tolist()
                },
                'rgb': {
                    'resolution': h5f['/ovc/rgb/calib/resolution'][:].tolist(),
                    'T_to_prophesee_left': h5f['/ovc/rgb/calib/T_to_prophesee_left'][:].tolist(),
                    'instrinsics': h5f['/ovc/rgb/calib/intrinsics'][:].tolist(),
                    'distortion_coeffs': h5f['/ovc/rgb/calib/distortion_coeffs'][:].tolist()
                }
            },
            'prophesee': {
                'left': {
                    'resolution': h5f['/prophesee/left/calib/resolution'][:].tolist(),
                    'T_to_prophesee_left': h5f['/prophesee/left/calib/T_to_prophesee_left'][:].tolist(),
                    'instrinsics': h5f['/prophesee/left/calib/intrinsics'][:].tolist(),
                    'distortion_coeffs': h5f['/prophesee/left/calib/distortion_coeffs'][:].tolist()
                },
                'right': {
                    'resolution': h5f['/prophesee/right/calib/resolution'][:].tolist(),
                    'T_to_prophesee_left': h5f['/prophesee/right/calib/T_to_prophesee_left'][:].tolist(),
                    'instrinsics': h5f['/prophesee/right/calib/intrinsics'][:].tolist(),
                    'distortion_coeffs': h5f['/prophesee/right/calib/distortion_coeffs'][:].tolist()
                }
            }
        }
        with open(f"{root}/cam_to_cam.yaml", 'w') as yamlf:
            yaml.dump(calib_data, yamlf)


# Read all images in /ovc/left/data first to get index we need
def readImage(args):
    GreyLeftSave_dir = f"{args.output}/Images/{args.seq_name}/left"
    os.makedirs(GreyLeftSave_dir, exist_ok=True)

    with h5py.File(args.data_path, 'r') as h5f:
        I_ts = h5f['/ovc/ts'][:]
        I_idx = list(range(0, len(I_ts), 1))
        with open(f"{args.output}/Images/{args.seq_name}/timestamps.txt", 'w') as f:
            for idx, timestamp in zip(I_idx, I_ts):
                f.write(f'{idx:08}, {timestamp}\n')

        # Save Left gray Image
        I_left_list = h5f['/ovc/left/data']
        image_num, width, height, channel = I_left_list.shape
        for idx in tqdm(range(image_num)):
            I_left = I_left_list[idx]
            I_left_pil = Image.fromarray(I_left.squeeze())
            I_left_pil.save(f"{GreyLeftSave_dir}/{idx:08}.png")


def readDepth(args):
    box_path = f"{args.output}/Boxes/{args.seq_name}"
    pd.read_csv(f"{box_path}/boxes.txt", sep=' ', header=None, names=['idx', 'ts', 'xmin', 'ymin', 'xmax', 'ymax'])
    with h5py.File(args.depth_gt_path, 'r') as h5f:
        aa = 1





if __name__ == "__main__":
    file_name = '/media/jhang/Elements/Public_Dataset/M3ED/Hdf5/car_urban_day_penno_big_loop_data.h5'




    with h5py.File("/media/jhang/Elements/Public_Dataset/M3ED/Hdf5/car_urban_day_penno_big_loop_data.h5", 'r') as h5f:
        aa = h5f['/ouster/data'][0]










    root_dir = '/home/jhang/workspace/STRTTC_RAL2023/code/STRTTC_matlab/dataset/real/M3ED'

    seq_list = ['car_urban_day_city_hall_3',
                'car_urban_day_ucity_small_loop_1',
                'car_urban_day_rittenhouse_1',
                'car_urban_day_ucity_small_loop_3'
    ]
    depth_gt_list = [
        '/media/jhang/Elements/Public_Dataset/M3ED/car_urban_day_city_hall_data.h5',
        '/media/jhang/Elements/Public_Dataset/M3ED/car_urban_day_ucity_small_loop_depth_gt.h5',
        '/media/jhang/Elements/Public_Dataset/M3ED/car_urban_day_rittenhouse_depth_gt.h5',
        '/media/jhang/Elements/Public_Dataset/M3ED/car_urban_day_ucity_small_loop_depth_gt.h5'
    ]
    for seq_id in range(len(seq_list)):
        data_dir = f"{root_dir}/{seq_list[seq_id]}"
        # rgb_ts_df = pd.read_csv(f"{data_dir}/rgb/frames/timestamps.txt", sep=' ', header=None, names=['frame_id', 'ts'])
        # rgb_ts_start = rgb_ts_df['ts'].iloc[0]
        # rgb_ts_end = rgb_ts_df['ts'].iloc[-1]
        depth_image_dir = f"{data_dir}/depth_image"
        os.makedirs(depth_image_dir, exist_ok=True)
        boxes_df = pd.read_csv(f"{data_dir}/bbox/bbox.txt", sep=' ', header=None, names=['frame_id', 'ts', 'xmin', 'ymin', 'xmax', 'ymax', 'blank'])
        boxes_ts_start = boxes_df['ts'].iloc[0]
        boxes_ts_end = boxes_df['ts'].iloc[-1]

        # Select depth ts from depth_gt.h5 file in the same time range as rgb
        with h5py.File(depth_gt_list[seq_id], 'r') as h5f:
            ts_list = h5f['/ouster/ts_end'][:]
            ts_select_index = np.where((ts_list >= boxes_ts_start) & (ts_list <= boxes_ts_end))[0]
            for depth_id in ts_select_index:
                depth_ts = ts_list[depth_id]
                # find row index of boxes_df that is closest to depth_ts
                diff = np.abs(boxes_df['ts'] - depth_ts)
                closest_index = np.argmin(diff)
                box_row = boxes_df.iloc[closest_index]
                diff_t = np.abs(depth_ts - box_row['ts'])
                # get depth image index and plot bbox on it
                legacy_data = h5f['/ouster/data'][depth_id]

    rgb_ts = pd.read_csv()


    with h5py.File(depth_h5, 'r') as h5f:
        depth_ts = h5f['/ts'][:]




        ts_map_prophesee_left = h5f['/ts_map_prophesee_left'][:]
        ts = h5f['/ts'][:]

        depth_prophesee_left = h5f['/depth/prophesee/left'][1]
        aa = 1





    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default='/mnt/disk1/M3ED/out')

    parser.add_argument("--data_path",
                        default="/media/jhang/Elements/Public_Dataset/M3ED/car_urban_day_ucity_small_loop_data.h5")
    parser.add_argument("--depth_gt_path",
                        default="/media/jhang/Elements/Public_Dataset/M3ED/car_urban_day_city_hall_depth_gt.h5")
    parser.add_argument("--seq_name", default='car_urban_day_ucity_small_loop')
    parser.add_argument("--events_range", default=(1, 2))
    args = parser.parse_args()

    export_all_image = False
    get_selected_seq = False
    get_depth = True

    with h5py.File(args.depth_gt_path, 'r') as h5f:
        aa = 1

    if export_all_image:
        readImage(args)

    if get_selected_seq:
        seq_pair = [(6330, 6445),
                    (9405, 9540),
                    (15190, 15301),
                    (16751, 16801)]
        for i in range(len(seq_pair)):
            args.events_range = seq_pair[i]
            get_strttc_data(args)
            print(f'done, {seq_pair[i]}')

    if get_depth:
        readDepth(args)



