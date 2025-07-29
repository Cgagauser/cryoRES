import torch
import numpy as np
import os
import json
from copy import deepcopy
import mrcfile
from scipy.ndimage import zoom
from Bio import PDB

def resize_3d_data(data, target_shape):
    zoom_factors = (
        target_shape[0] / data.shape[0],
        target_shape[1] / data.shape[1],
        target_shape[2] / data.shape[2]
    )
    print("zoom_factors:", zoom_factors)
    resized_data = zoom(data, zoom_factors, order=3) 
    return resized_data, zoom_factors

def save_resized_mrc(data, output_path, original_header=None, zoom_factors=None):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    data = data.astype(np.float32) 
    
    with mrcfile.new(output_path, overwrite=True) as mrc:
        mrc.set_data(data)
        
        # 如果有原始头信息,保留关键的坐标系统信息
        if original_header is not None:
            # 保持原始的原点信息
            mrc.header.origin.x = original_header.origin.x
            mrc.header.origin.y = original_header.origin.y
            mrc.header.origin.z = original_header.origin.z
            # mrc.header.origin.x = 100000000000
            # mrc.header.origin.y = 100000000000000
            # mrc.header.origin.z = 100000000
            # 保持原始的起始索引
            mrc.header.nxstart = original_header.nxstart
            mrc.header.nystart = original_header.nystart
            mrc.header.nzstart = original_header.nzstart
            
            # 重要：保持原始的cell dimensions（物理尺寸）
            if zoom_factors is not None:
                mrc.header.cella.x = original_header.cella.x * zoom_factors[2]
                mrc.header.cella.y = original_header.cella.y * zoom_factors[1]
                mrc.header.cella.z = original_header.cella.z * zoom_factors[0]
            
            # 保持原始的角度信息
            mrc.header.cellb.alpha = original_header.cellb.alpha
            mrc.header.cellb.beta = original_header.cellb.beta
            mrc.header.cellb.gamma = original_header.cellb.gamma
            
            # 保持原始的轴顺序
            mrc.header.mapc = original_header.mapc
            mrc.header.mapr = original_header.mapr
            mrc.header.maps = original_header.maps
        
        # 更新数据相关的信息（但不覆盖cell dimensions）
        mrc.header.nx = data.shape[2]
        mrc.header.ny = data.shape[1]
        mrc.header.nz = data.shape[0]
        mrc.header.mx = data.shape[2]
        mrc.header.my = data.shape[1]
        mrc.header.mz = data.shape[0]
        
        # 更新统计信息
        mrc.header.dmin = data.min()
        mrc.header.dmax = data.max()
        mrc.header.dmean = data.mean()

def get_data(dir_path):
    protein_data, seq, chain_index = None, None, None
    for file in os.listdir(dir_path):
        if file.endswith('.mrc'):
            dm_path = os.path.join(dir_path, file)
            try:
                with mrcfile.open(dm_path, mode='r') as p_map:
                    protein_data = deepcopy(p_map.data)
                    original_header = deepcopy(p_map.header)  # 保存原始头信息
                
                # 打印原始信息
                print(f"Original shape: {protein_data.shape}")
                print(f"Original voxel size: ({original_header.cella.x/original_header.mx:.3f}, "
                      f"{original_header.cella.y/original_header.my:.3f}, "
                      f"{original_header.cella.z/original_header.mz:.3f})")
                print(f"Original cell dimensions: ({original_header.cella.x:.3f}, "
                      f"{original_header.cella.y:.3f}, {original_header.cella.z:.3f})")
                
                protein_data, zoom_factors = resize_3d_data(protein_data, [360, 360, 360])
                
                # 传递原始头信息和zoom factors
                save_resized_mrc(protein_data, dir_path + "/resize360.map", 
                               original_header, zoom_factors)
                
                # 验证新文件
                with mrcfile.open(dir_path + "/resize360.map", mode='r') as new_map:
                    new_header = new_map.header
                    print(f"\nNew shape: {new_map.data.shape}")
                    print(f"New voxel size: ({new_header.cella.x/new_header.mx:.3f}, "
                          f"{new_header.cella.y/new_header.my:.3f}, "
                          f"{new_header.cella.z/new_header.mz:.3f})")
                    print(f"New cell dimensions: ({new_header.cella.x:.3f}, "
                          f"{new_header.cella.y:.3f}, {new_header.cella.z:.3f})")
  
                # PDB坐标缩放保持不变
                with open("/ziyingz/Programs/E3-CryoFold/testtemp/7XXF_4_H/7XXF_4_H.pdb") as fin, \
                     open("/ziyingz/Programs/E3-CryoFold/testtemp/7XXF_4_H/scaled", "w") as fout:
                    for line in fin:
                        if line.startswith("ATOM"):
                            # x = (float(line[30:38])-original_header.origin.x) * zoom_factors[2]
                            # y = (float(line[38:46])-original_header.origin.y) * zoom_factors[1]
                            # z = (float(line[46:54])-original_header.origin.z) * zoom_factors[0]
                            x = float(line[30:38]) * zoom_factors[2]
                            y = float(line[38:46]) * zoom_factors[1]
                            z = float(line[46:54]) * zoom_factors[0]
                            fout.write(f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}")
                        else:
                            fout.write(line)     
            except Exception as e:
                print(f"Error loading or processing the density map: {e}")
                raise
    return protein_data
    
    
get_data("/ziyingz/Programs/E3-CryoFold/testtemp/7XXF_4_H")