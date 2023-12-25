import yaml
import argparse
import h5py
from PIL import Image
import numpy as np
import os


def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)
    
    
def copy_group_datasets(source_group, dest_group):
    """
    Copy all datasets within source_group to dest_group
    """
    # Search for datasets in source_group
    for dset_name, dset in source_group.items():
        if isinstance(dset, h5py.Dataset):
            dest_group.copy(dset, dset_name)
            

def find_index_in_array(array: np.ndarray, search_value, left=True):
    """
    Find the index of the value in the array.
    If left is True, find the leftmost index.
    """
    if left:
        idx = np.searchsorted(array, search_value, side='left')
        if idx > 0:
            idx = idx - 1
            value = array[idx]
            print(f"Search Value: {search_value}, Left Value: {value}, index of Left Value: {idx}")
        else:
            raise ValueError(f"Search Value: {search_value} is smaller than the smallest value in the array.")
    else:
        idx = np.searchsorted(array, search_value, side='right')
        if idx < len(array):
            value = array[idx]
            print(f"Search Value: {search_value}, Right Value: {value}, index of Right Value: {idx}")
        else:
            raise ValueError(f"Search Value: {search_value} is larger than the largest value in the array.")
        
    return idx, value
    

def copy_dataset_by_idxs(origin_hf5, new_hf5, dataset_name, start_idx, stop_idx, copy_chunks=False):
    if copy_chunks:
        new_hf5.create_dataset(dataset_name, 
                           data=origin_hf5[dataset_name][start_idx:stop_idx, ...],
                           dtype=origin_hf5[dataset_name].dtype,
                           compression=origin_hf5[dataset_name].compression,
                           chunks=origin_hf5[dataset_name].chunks)
    else:
        new_hf5.create_dataset(dataset_name, 
                            data=origin_hf5[dataset_name][start_idx:stop_idx, ...], 
                            dtype=origin_hf5[dataset_name].dtype,
                            compression=origin_hf5[dataset_name].compression)

    return new_hf5


def convert_to_milliseconds(four_digit_number):
    # 分割数字
    minutes = four_digit_number // 100  # 获取前两位
    seconds = four_digit_number % 100   # 获取后两位

    # 将分钟和秒转换为微秒
    total_milliseconds = (minutes * 60 + seconds) * 1000
    return total_milliseconds


def main(args):
    
    config = load_config(args.config)
    
    if not os.path.exists(config['save_root']):
        os.makedirs(config['save_root'], exist_ok=True)
    for file in config['files']:
        ts_start_ms = convert_to_milliseconds(file['start_time'])
        ts_stop_ms = convert_to_milliseconds(file['stop_time'])
        
        originPath_hf5 = os.path.join(config['hdf5_root'], file['file_name'])
        outPath_hf5 = os.path.join(config['save_root'], file['save_name'])
        
        print(f"Processing {originPath_hf5}, time: {file['start_time']}--{file['stop_time']}, save to {outPath_hf5}")
        
        if os.path.exists(outPath_hf5):
            os.remove(outPath_hf5)
        
        with h5py.File(originPath_hf5, 'r') as origin_hf5:
            with h5py.File(outPath_hf5, 'w') as new_hf5:
                
                new_hf5.attrs['raw_file_name'] = file['file_name']
                new_hf5.attrs['start_time_ms'] = ts_start_ms
                new_hf5.attrs['stop_time_ms'] = ts_stop_ms
                
                # Copy data we need
                # OVC
                ovc_start_idx, _ = find_index_in_array(origin_hf5['/ovc/ts'][:], ts_start_ms*1e3, left=True)
                ovc_stop_idx, _   = find_index_in_array(origin_hf5['/ovc/ts'][:], ts_stop_ms*1e3, left=False)
                
                
                copy_group_datasets(origin_hf5['/ovc/left/calib'], new_hf5.create_group('/ovc/left/calib'))
                new_hf5 = copy_dataset_by_idxs(origin_hf5, new_hf5, '/ovc/ts', 
                                        start_idx=ovc_start_idx, stop_idx=ovc_stop_idx)
                new_hf5 = copy_dataset_by_idxs(origin_hf5, new_hf5, '/ovc/ts_map_prophesee_left_t', 
                                        start_idx=ovc_start_idx, stop_idx=ovc_stop_idx)
                new_hf5 = copy_dataset_by_idxs(origin_hf5, new_hf5, '/ovc/left/data',
                                        start_idx=ovc_start_idx, stop_idx=ovc_stop_idx, copy_chunks=True)
                
                # Prophesee Left
                # Find the index in the event stream that correlates with the image time
                eventLeft_start_idx = origin_hf5['/ovc/ts_map_prophesee_left_t'][ovc_start_idx]
                eventLeft_stop_idx = origin_hf5['/ovc/ts_map_prophesee_left_t'][ovc_stop_idx]
                                                     
                
                copy_group_datasets(origin_hf5['/prophesee/left/calib'], new_hf5.create_group('/prophesee/left/calib'))
                
                new_hf5 = copy_dataset_by_idxs(origin_hf5, new_hf5, '/prophesee/left/p', 
                                        start_idx=eventLeft_start_idx, stop_idx=eventLeft_stop_idx, copy_chunks=True)
                new_hf5 = copy_dataset_by_idxs(origin_hf5, new_hf5, '/prophesee/left/t',
                                        start_idx=eventLeft_start_idx, stop_idx=eventLeft_stop_idx, copy_chunks=True)
                new_hf5 = copy_dataset_by_idxs(origin_hf5, new_hf5, '/prophesee/left/x',
                                        start_idx=eventLeft_start_idx, stop_idx=eventLeft_stop_idx, copy_chunks=True)
                new_hf5 = copy_dataset_by_idxs(origin_hf5, new_hf5, '/prophesee/left/y',
                                        start_idx=eventLeft_start_idx, stop_idx=eventLeft_stop_idx, copy_chunks=True) 
                
                
                # ouster
                lidar_start_idx, _ = find_index_in_array(origin_hf5['/ouster/ts_start'][:], ts_start_ms*1e3, left=True)
                lidar_stop_idx, _  = find_index_in_array(origin_hf5['/ouster/ts_start'][:], ts_stop_ms*1e3, left=False)
                
                copy_group_datasets(origin_hf5['/ouster/calib'], new_hf5.create_group('/ouster/calib'))
                
                new_hf5.copy(origin_hf5['/ouster/metadata'], 'ouster/metadata')
                
                new_hf5 = copy_dataset_by_idxs(origin_hf5, new_hf5, '/ouster/ts_start', 
                                        start_idx=lidar_start_idx, stop_idx=lidar_stop_idx)
                new_hf5 = copy_dataset_by_idxs(origin_hf5, new_hf5, '/ouster/ts_end', 
                                        start_idx=lidar_start_idx, stop_idx=lidar_stop_idx)
                new_hf5 = copy_dataset_by_idxs(origin_hf5, new_hf5, '/ouster/data', 
                                        start_idx=lidar_start_idx, stop_idx=lidar_stop_idx, copy_chunks=True)
            

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Process Hdf5 files from m3ed.')
    parser.add_argument('--config', help='Path to the YAML config file', default='mycode/config/config.yaml')
    args = parser.parse_args()
    
    main(args)
