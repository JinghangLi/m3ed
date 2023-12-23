import os
import h5py
import argparse
from tqdm import tqdm
from PIL import Image
import yaml
import pandas as pd
import numpy as np
import cv2 as cv
import open3d as o3d
from matplotlib import pyplot as plt
import json


def get_camera_param(h5f_intrinsics, h5f_distortion_coeffs):
    K_list = h5f_intrinsics
    K_array = np.eye(3)  # 左相机内参矩阵
    K_array[0, 0] = K_list[0]
    K_array[1, 1] = K_list[1]
    K_array[0, 2] = K_list[2]
    K_array[1, 2] = K_list[3]

    dist = h5f_distortion_coeffs
    distCoeffs_array = np.array([dist[0], dist[1], dist[2], dist[3], 0])

    return K_array, distCoeffs_array


def transform_point_cloud(point_cloud: np.ndarray, transformation_matrix: np.ndarray) -> np.ndarray:
    """
    Transform the given point cloud with the given transformation matrix.

    :param point_cloud: the point cloud to transform, as a numpy array
    :param transformation_matrix: the transformation matrix, as a 4x4 numpy array
    :return: the transformed point cloud, as a numpy array
    """
    # Convert the point cloud to a numpy array
    point_cloud_array = point_cloud

    # Add a fourth dimension of ones to the point cloud array
    point_cloud_array_homogeneous = np.hstack((point_cloud_array, np.ones((point_cloud_array.shape[0], 1))))

    # Apply the transformation matrix to the point cloud
    transformed_point_cloud_array_homogeneous = np.dot(transformation_matrix, point_cloud_array_homogeneous.T).T

    # Convert the transformed point cloud array back to a pcl.PointCloud
    transformed_point_cloud = transformed_point_cloud_array_homogeneous[:, :3]

    return transformed_point_cloud



def get_strttc_data(args):
    save_path = f'/home/jhang/workspace/STRTTC_RAL2023/code/STRTTC_matlab/dataset/real/M3ED_new/{args.seq_name}'
    root = save_path
    name_i = 1
    while os.path.exists(root):
        root = f"{save_path}_{name_i}"
        name_i += 1

    e_save = f"{root}/txt"
    rgb_save = f"{root}/rgb/frames"
    right_gray_save = f"{root}/left/frames"
    e_image_save = f"{root}/txt_image"

    os.makedirs(e_save, exist_ok=True)
    os.makedirs(rgb_save, exist_ok=True)
    os.makedirs(e_image_save, exist_ok=True)
    os.makedirs(right_gray_save, exist_ok=True)

    # 创建 SGBM 对象 for depth calculation
    window_size = 3
    min_disp = 16
    num_disp = 112 - min_disp
    stereo = cv.StereoSGBM_create(minDisparity=min_disp,
                                  numDisparities=num_disp,
                                  blockSize=16,
                                  P1=8 * 3 * window_size ** 2,
                                  P2=32 * 3 * window_size ** 2,
                                  disp12MaxDiff=1,
                                  uniquenessRatio=10,
                                  speckleWindowSize=100,
                                  speckleRange=32
                                  )

    with h5py.File(args.data_path, 'r') as h5f:
        with open(f"{rgb_save}/timestamps.txt", 'w') as f:
            ts_start = h5f['/ovc/ts'][args.events_range[0]]
            ts_end = h5f['/ovc/ts'][args.events_range[1]]
            for idx in tqdm(range(args.events_range[0], args.events_range[1])):
                name_id = "{:0>10d}".format(int(idx))

                lidar = h5f['/ouster/data'][0]
                metadata_str = h5f['/ouster/metadata'][...].item().decode('utf-8')
                metadata = json.loads(metadata_str)
                beam_altitude_angles = np.array(metadata['beam_altitude_angles'])
                beam_azimuth_angles = np.array(metadata['beam_azimuth_angles'])
                pixel_shift_by_row = np.array(metadata['data_format']['pixel_shift_by_row'])
                columns_per_frame = np.array(metadata['data_format']['columns_per_frame'])

                data = lidar
                # Initialize the (x, y, z) coordinate arrays
                x = np.zeros((len(data),))
                y = np.zeros((len(data),))
                z = np.zeros((len(data),))

                # Loop over each row in the data
                for i in range(len(data)):
                    # Calculate the beam angle for the current row
                    beam_altitude_angle = beam_altitude_angles[i]
                    beam_azimuth_angle = beam_azimuth_angles[i]

                    # Calculate the (x, y, z) coordinates for the current row
                    x[i] = np.cos(beam_azimuth_angle) * data['distance'][i]
                    y[i] = np.sin(beam_azimuth_angle) * data['distance'][i]
                    z[i] = beam_altitude_angle - 90

                # Stack x, y, z to create a 3D point cloud
                point_cloud_array = np.dstack((x, y, z))

                # 创建一个PointCloud对象
                point_cloud = o3d.geometry.PointCloud()
                # 将numpy数组转换为PointCloud
                point_cloud.points = o3d.utility.Vector3dVector(point_cloud_array)

                # 使用open3d来显示点云
                o3d.visualization.draw_geometries([point_cloud])


                gray_left = h5f['/ovc/left/data'][idx]
                I_left_pil_left = Image.fromarray(gray_left.squeeze())
                I_left_pil_left.save(f"{rgb_save}/frame_{name_id}.png")

                gray_right = h5f['/ovc/right/data'][idx]
                I_left_pil_right = Image.fromarray(gray_right.squeeze())
                I_left_pil_right.save(f"{right_gray_save}/frame_{name_id}.png")

                # if idx == args.events_range[1]-1:
                #     I_left_pil.save(f"/mnt/disk1/M3ED/sample/{args.seq_name}_{name_id}.png")
                e_idx_start = h5f['/ovc/ts_map_prophesee_left_t'][idx - 1]
                e_idx_end = h5f['/ovc/ts_map_prophesee_left_t'][idx]

                left_events_x = h5f['/prophesee/left/x'][e_idx_start:e_idx_end]
                left_events_y = h5f['/prophesee/left/y'][e_idx_start:e_idx_end]
                left_events_t = h5f['/prophesee/left/t'][e_idx_start:e_idx_end]
                left_events_p = h5f['/prophesee/left/p'][e_idx_start:e_idx_end]

                # format for STRTTC
                left_events_t = left_events_t / 1e9
                left_events_p = np.where(left_events_p == 0, -1, left_events_p)

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

                ts = h5f['/ovc/ts'][idx] / 1e9
                f.write(f'{idx} {ts}\n')

                # ----------------------------------
                # depth calculation
                imgL = cv.normalize(gray_left, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
                imgR = cv.normalize(gray_right, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

                K_left, distCoeffs_left = get_camera_param(h5f['/ovc/left/calib/intrinsics'][:].tolist(),
                                                              h5f['/ovc/left/calib/distortion_coeffs'][:].tolist())
                K_right, distCoeffs_right = get_camera_param(h5f['/ovc/right/calib/intrinsics'][:].tolist(),
                                                                h5f['/ovc/right/calib/distortion_coeffs'][:].tolist())
                T_eventL_grayL = np.array(h5f['/ovc/left/calib/T_to_prophesee_left'][:].tolist())
                T_eventL_grayR = np.array(h5f['/ovc/right/calib/T_to_prophesee_left'][:].tolist())
                T_grayR_grayL = np.matmul(np.linalg.inv(T_eventL_grayR), T_eventL_grayL)
                R_grayR_grayL = T_grayR_grayL[:3, :3]
                t_grayR_grayL = T_grayR_grayL[:3, 3]
                imageSize = np.array(h5f['/ovc/left/calib/resolution'][:].tolist())

                R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv.stereoRectify(K_left, distCoeffs_left, K_right, distCoeffs_right, imageSize, R_grayR_grayL, t_grayR_grayL)
                map1, map2 = cv.initUndistortRectifyMap(K_left, distCoeffs_left, R1, P1, imageSize, 5)
                map3, map4 = cv.initUndistortRectifyMap(K_right, distCoeffs_right, R2, P2, imageSize, 5)
                imgL_rect = cv.remap(imgL, map1, map2, cv.INTER_LINEAR)
                imgR_rect = cv.remap(imgR, map3, map4, cv.INTER_LINEAR)

                disparity = stereo.compute(imgL_rect, imgR_rect).astype(np.float32) / 16.0

                plt.figure(figsize=(10, 7))
                plt.imshow(disparity, 'gray')
                plt.colorbar()
                plt.show()

                cloud_stereo = cv.reprojectImageTo3D(disparity, Q)
                cloud_stereo_points = cloud_stereo.reshape(-1, 3)

                points_camRect1 = cloud_stereo_points
                points_eventL = transform_point_cloud(points_camRect1, T_eventL_grayL)

                # 创建一个PointCloud对象
                point_cloud = o3d.geometry.PointCloud()
                # 将numpy数组转换为PointCloud
                point_cloud.points = o3d.utility.Vector3dVector(points_eventL)

                # 使用open3d来显示点云
                o3d.visualization.draw_geometries([point_cloud])

                aa = 1





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

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default='/mnt/disk1/M3ED/out')

    parser.add_argument("--data_path",
                        default="/media/jhang/Elements/Public_Dataset/M3ED/car_urban_day_city_hall_data.h5")
    parser.add_argument("--depth_gt_path",
                        default="/media/jhang/Elements/Public_Dataset/M3ED/car_urban_day_city_hall_depth_gt.h5")
    parser.add_argument("--seq_name", default='car_urban_day_city_hall')
    parser.add_argument("--events_range", default=(1, 2))
    args = parser.parse_args()

    export_all_image = False
    get_selected_seq = True
    get_depth = False

    if export_all_image:
        readImage(args)

    if get_selected_seq:
        # seq_pair = [(6330, 6445),
        #             (9405, 9540),
        #             (15190, 15301),
        #             (16751, 16801)]
        seq_pair = [(6279, 6374)]
        for i in range(len(seq_pair)):
            args.events_range = seq_pair[i]
            get_strttc_data(args)
            print(f'done, {seq_pair[i]}')

    if get_depth:
        readDepth(args)



