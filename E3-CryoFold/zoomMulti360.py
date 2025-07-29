import os
import glob
import numpy as np
import mrcfile
import torch
from scipy.ndimage import zoom
from copy import deepcopy
from multiprocessing import Pool
import argparse
from tqdm import tqdm
import time

def resize_3d_data(data, target_shape):
    """调整3D数据尺寸"""
    zoom_factors = (
        target_shape[0] / data.shape[0],
        target_shape[1] / data.shape[1],
        target_shape[2] / data.shape[2]
    )
    print(f"zoom_factors: {zoom_factors}")
    resized_data = zoom(data, zoom_factors, order=3) 
    return resized_data, zoom_factors

def save_resized_mrc(data, output_path, original_header=None, zoom_factors=None):
    """保存调整尺寸后的MRC文件"""
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
            
            # 保持原始的起始索引
            mrc.header.nxstart = original_header.nxstart
            mrc.header.nystart = original_header.nystart
            mrc.header.nzstart = original_header.nzstart
            
            # 重要：按照zoom factors缩放cell dimensions
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
        
        # 更新数据相关的信息
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

def process_single_folder(folder_path):
    """处理单个文件夹的函数"""
    folder_name = os.path.basename(folder_path)
    print(f"[{os.getpid()}] 开始处理文件夹: {folder_name}")
    
    try:
        # 查找MRC文件
        mrc_files = glob.glob(os.path.join(folder_path, "*.mrc"))
        if not mrc_files:
            print(f"[{os.getpid()}] 警告: {folder_name} 中没有找到.mrc文件")
            return {'status': 'no_mrc', 'folder': folder_name}
        
        # 查找PDB文件  
        pdb_files = glob.glob(os.path.join(folder_path, "*.pdb"))
        if not pdb_files:
            print(f"[{os.getpid()}] 警告: {folder_name} 中没有找到.pdb文件")
            return {'status': 'no_pdb', 'folder': folder_name}
        
        # 处理第一个找到的MRC文件
        mrc_path = mrc_files[0]
        pdb_path = pdb_files[0]
        
        # 检查输出文件是否已存在
        resize_output = os.path.join(folder_path, "resize360.map")
        scaled_output = os.path.join(folder_path, "scaled")
        
        if os.path.exists(resize_output) and os.path.exists(scaled_output):
            print(f"[{os.getpid()}] 跳过 {folder_name} (文件已存在)")
            return {'status': 'skipped', 'folder': folder_name}
        
        # 读取并处理MRC文件
        with mrcfile.open(mrc_path, mode='r') as p_map:
            protein_data = deepcopy(p_map.data)
            original_header = deepcopy(p_map.header)
        
        print(f"[{os.getpid()}] {folder_name} - 原始形状: {protein_data.shape}")
        
        # 调整尺寸到360x360x360
        resized_data, zoom_factors = resize_3d_data(protein_data, [360, 360, 360])
        
        # 保存调整尺寸后的MRC文件
        save_resized_mrc(resized_data, resize_output, original_header, zoom_factors)
        
        # 生成缩放后的PDB文件
        with open(pdb_path, 'r') as fin, open(scaled_output, 'w') as fout:
            for line in fin:
                if line.startswith("ATOM"):
                    x = (float(line[30:38]) - original_header.origin.x) * zoom_factors[2]
                    y = (float(line[38:46]) - original_header.origin.y) * zoom_factors[1]
                    z = (float(line[46:54]) - original_header.origin.z) * zoom_factors[0]
                    fout.write(f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}")
                else:
                    fout.write(line)
        
        # 验证输出文件
        with mrcfile.open(resize_output, mode='r') as new_map:
            new_header = new_map.header
            print(f"[{os.getpid()}] {folder_name} - 新形状: {new_map.data.shape}")
            print(f"[{os.getpid()}] {folder_name} - 新voxel size: "
                  f"({new_header.cella.x/new_header.mx:.3f}, "
                  f"{new_header.cella.y/new_header.my:.3f}, "
                  f"{new_header.cella.z/new_header.mz:.3f})")
        
        print(f"[{os.getpid()}] ✅ 成功处理: {folder_name}")
        return {'status': 'success', 'folder': folder_name}
        
    except Exception as e:
        print(f"[{os.getpid()}] ❌ 处理 {folder_name} 时出错: {str(e)}")
        return {'status': 'error', 'folder': folder_name, 'error': str(e)}

def get_subfolders(root_dir):
    """获取所有子文件夹"""
    subfolders = []
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            subfolders.append(item_path)
    return subfolders

def main():
    parser = argparse.ArgumentParser(description='并行处理MRC和PDB文件')
    parser.add_argument('--root_dir', type=str, required=True, 
                       help='包含多个子文件夹的根目录')
    parser.add_argument('--num_cores', type=int, default=15,
                       help='并行处理的核心数 (默认: 15)')
    parser.add_argument('--chunk_size', type=int, default=15,
                       help='每次处理的文件夹数量 (默认: 1)')
    
    args = parser.parse_args()
    
    # 检查根目录是否存在
    if not os.path.exists(args.root_dir):
        print(f"错误: 根目录 {args.root_dir} 不存在")
        return
    
    # 获取所有子文件夹
    subfolders = get_subfolders(args.root_dir)
    print(f"找到 {len(subfolders)} 个子文件夹")
    
    if not subfolders:
        print("没有找到子文件夹")
        return
    
    # 显示前几个文件夹作为示例
    print("前5个文件夹:")
    for i, folder in enumerate(subfolders[:5]):
        print(f"  {i+1}. {os.path.basename(folder)}")
    if len(subfolders) > 5:
        print(f"  ... 还有 {len(subfolders)-5} 个文件夹")
    
    # 确认是否继续
    response = input(f"\n是否使用 {args.num_cores} 个核心并行处理这些文件夹? (y/n): ")
    if response.lower() != 'y':
        print("取消处理")
        return
    
    # 开始并行处理
    start_time = time.time()
    print(f"\n开始并行处理，使用 {args.num_cores} 个核心...")
    
    # 使用进程池并行处理
    with Pool(processes=args.num_cores) as pool:
        # 使用tqdm显示进度条
        results = list(tqdm(
            pool.imap(process_single_folder, subfolders, chunksize=args.chunk_size),
            total=len(subfolders),
            desc="处理进度"
        ))
    
    end_time = time.time()
    
    # 统计结果
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = sum(1 for r in results if r['status'] == 'error')
    skipped_count = sum(1 for r in results if r['status'] == 'skipped')
    no_mrc_count = sum(1 for r in results if r['status'] == 'no_mrc')
    no_pdb_count = sum(1 for r in results if r['status'] == 'no_pdb')
    
    print(f"\n" + "="*50)
    print("处理完成!")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print(f"总文件夹数: {len(subfolders)}")
    print(f"成功处理: {success_count}")
    print(f"跳过 (已存在): {skipped_count}")
    print(f"错误: {error_count}")
    print(f"缺少MRC文件: {no_mrc_count}")
    print(f"缺少PDB文件: {no_pdb_count}")
    
    # 显示错误详情
    if error_count > 0:
        print(f"\n错误详情:")
        for result in results:
            if result['status'] == 'error':
                print(f"  - {result['folder']}: {result['error']}")

def simple_batch_process(root_dir, num_cores=15):
    """简化版本的批量处理函数，可以直接调用"""
    subfolders = get_subfolders(root_dir)
    print(f"找到 {len(subfolders)} 个子文件夹，开始处理...")
    
    start_time = time.time()
    
    with Pool(processes=num_cores) as pool:
        results = list(tqdm(
            pool.imap(process_single_folder, subfolders, chunksize=1),
            total=len(subfolders),
            desc="处理进度"
        ))
    
    end_time = time.time()
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"处理完成! 成功: {success_count}/{len(subfolders)}, 耗时: {end_time - start_time:.2f}秒")
    return results

if __name__ == "__main__":
    # 方式1: 命令行使用
    main()
    
    # 方式2: 直接在代码中调用 (注释掉上面的main()，取消注释下面的代码)
    # root_directory = "/path/to/your/root/directory"
    # results = simple_batch_process(root_directory, num_cores=15)