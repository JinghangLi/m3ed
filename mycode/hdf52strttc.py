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
            
            
def create_datasets(origin_hf5, new_hf5, dataset_name, copy_idx, copy_chunks=False):
    if copy_chunks:
        new_hf5.create_dataset(dataset_name, 
                           data=origin_hf5[dataset_name][copy_idx, ...], 
                           dtype=origin_hf5[dataset_name].dtype,
                           compression=origin_hf5[dataset_name].compression,
                           chunks=origin_hf5[dataset_name].chunks)
    else:
    
        new_hf5.create_dataset(dataset_name, 
                            data=origin_hf5[dataset_name][copy_idx, ...], 
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
        ts_end_ms = convert_to_milliseconds(file['end_time'])
        
        originPath_hf5 = os.path.join(config['hdf5_root'], file['file_name'])
        outPath_hf5 = os.path.join(config['save_root'], file['save_name'])
        
        print(f"Processing {originPath_hf5}, time: {file['start_time']}--{file['end_time']}, save to {outPath_hf5}")
        
        if os.path.exists(outPath_hf5):
            os.remove(outPath_hf5)
        
        with h5py.File(originPath_hf5, 'r') as origin_hf5:
            with h5py.File(outPath_hf5, 'w') as new_hf5:
                
                new_hf5.attrs['raw_file_name'] = file['file_name']
                new_hf5.attrs['start_time_ms'] = ts_start_ms
                new_hf5.attrs['end_time_ms'] = ts_end_ms
                
                # Copy data we need
                # OVC
                copy_group_datasets(origin_hf5['/ovc/left/calib'], new_hf5.create_group('/ovc/left/calib'))
                
                OVC_ts_idx_list = np.logical_and(origin_hf5['/ovc/ts'][:] >= ts_start_ms*1e3, 
                                            origin_hf5['/ovc/ts'][:] <= ts_end_ms*1e3)
                origin_hf5[dataset_name][copy_idx, ...]
                
                new_hf5 = create_datasets(origin_hf5, new_hf5, '/ovc/ts', OVC_ts_idx)
                new_hf5 = create_datasets(origin_hf5, new_hf5, '/ovc/ts_map_prophesee_left_t', OVC_ts_idx)
                new_hf5 = create_datasets(origin_hf5, new_hf5, '/ovc/left/data', OVC_ts_idx, copy_chunks=True)
                
                
                # Prophesee Left
                copy_group_datasets(origin_hf5['/prophesee/left/calib'], new_hf5.create_group('/prophesee/left/calib'))
                
                eventLeft_num_idx = origin_hf5['/ovc/ts_map_prophesee_left_t'][OVC_ts_idx]  
                new_hf5 = create_datasets(origin_hf5, new_hf5, '/prophesee/left/p', eventLeft_num_idx)
                new_hf5 = create_datasets(origin_hf5, new_hf5, '/prophesee/left/t', eventLeft_num_idx)
                new_hf5 = create_datasets(origin_hf5, new_hf5, '/prophesee/left/x', eventLeft_num_idx)
                new_hf5 = create_datasets(origin_hf5, new_hf5, '/prophesee/left/y', eventLeft_num_idx)
                
                
                
                
                
            
            
            



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Process Hdf5 files from m3ed.')
    parser.add_argument('--config', help='Path to the YAML config file', default='mycode/config/config.yaml')
    args = parser.parse_args()
    
    main(args)
