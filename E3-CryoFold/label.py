from Bio.PDB import PDBParser



# label.py
import numpy as np
import torch
import mrcfile
import os

def read_map_origin(map_path):
    """从MRC/MAP文件读取原点"""
    try:
        with mrcfile.open(map_path, mode='r') as mrc:
            # MRC文件中的原点信息
            origin = (mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z)
            return origin
    except Exception as e:
        print(f"读取密度图原点失败: {e}")
        return (0, 0, 0)  # 默认原点

# def get_ca_coords(pdb_path):
    # """从PDB文件获取CA坐标"""
    # ca_coords = []
    # try:
        # with open(pdb_path, 'r') as f:
            # for line in f:
                # if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                    # x = float(line[30:38])
                    # y = float(line[38:46])
                    # z = float(line[46:54])
                    # ca_coords.append({
                        # 'coord': np.array([x, y, z]),
                        # 'resid': int(line[22:26]),
                        # 'chain': line[21]
                    # })
    # except Exception as e:
        # print(f"读取PDB文件失败: {e}")
    # return ca_coords
def get_ca_coords(pdb_path):
    """从PDB文件获取CA坐标，正确处理多构象"""
    ca_coords = []
    seen_residues = set()  # 记录已经处理的残基
    
    try:
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                    resid = int(line[22:26])
                    chain = line[21]
                    altloc = line[16]  # 备选位置指示符（第17列）
                    
                    # 构建唯一的残基标识
                    residue_key = (chain, resid)
                    
                    # 如果这个残基已经处理过，跳过（只保留第一个构象）
                    if residue_key in seen_residues:
                        continue
                    
                    # 如果有备选位置，只取第一个（通常是'A'或空格）
                    if altloc not in [' ', 'A']:
                        continue
                    
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    
                    ca_coords.append({
                        'coord': np.array([x, y, z]),
                        'resid': resid,
                        'chain': chain
                    })
                    
                    seen_residues.add(residue_key)
                    
    except Exception as e:
        print(f"读取PDB文件失败: {e}")
    
    #print(f"读取到 {len(ca_coords)} 个唯一的CA原子")
    return ca_coords
def which_patch_pdb_ca(coord, origin=(0, 0, 0), grid_shape=(10, 10, 10), patch_size=36):
    """
    计算PDB坐标属于哪个patch
    coord: [x, y, z] (PDB坐标)
    origin: 密度图原点
    grid_shape: (Z, Y, X)
    patch_size: patch大小
    """
    # 将坐标转换到相对于密度图原点的坐标系
    relative_coord = coord - np.array(origin)
    
    # 计算patch索引
    i = int(relative_coord[2] // patch_size)    # z轴对应第一个维度
    j = int(relative_coord[1] // patch_size)    # y轴对应第二个维度
    k = int(relative_coord[0] // patch_size)    # x轴对应第三个维度
    
    # 边界检查
    i = np.clip(i, 0, grid_shape[0]-1)
    j = np.clip(j, 0, grid_shape[1]-1)
    k = np.clip(k, 0, grid_shape[2]-1)
    
    # 计算线性索引
    n = i * (grid_shape[1] * grid_shape[2]) + j * grid_shape[2] + k
    return n

def patch_id_to_center(patch_id, grid_shape, patch_size, origin):
    """将patch编号转换为中心坐标"""
    i = patch_id // (grid_shape[1] * grid_shape[2])
    j = (patch_id % (grid_shape[1] * grid_shape[2])) // grid_shape[2]
    k = patch_id % grid_shape[2]
    
    center_x = origin[0] + k * patch_size + patch_size / 2
    center_y = origin[1] + j * patch_size + patch_size / 2
    center_z = origin[2] + i * patch_size + patch_size / 2
    
    return np.array([center_x, center_y, center_z])

def generate_gaussian_probability(true_patch_id, grid_shape, patch_size, origin=(0, 0, 0), sigma=0.4):
    """为一个残基生成高斯概率分布"""
    total_patches = grid_shape[0] * grid_shape[1] * grid_shape[2]
    probabilities = np.zeros(total_patches)
    
    true_center = patch_id_to_center(true_patch_id, grid_shape, patch_size, origin)
    
    for patch_id in range(total_patches):
        patch_center = patch_id_to_center(patch_id, grid_shape, patch_size, origin)
        distance = np.linalg.norm(patch_center - true_center)
        probabilities[patch_id] = np.exp(-distance ** 2 / (2 * (sigma * patch_size) ** 2))
    
    # 归一化
    probabilities = probabilities / np.sum(probabilities)
    return probabilities

def generate_gaussian_labels(folder_path, cube_size=(360, 360, 360), patch_size=36, sigma=0.4):
    """
    为整个文件夹生成高斯标签
    
    参数:
        folder_path: 包含PDB和密度图的文件夹路径
        cube_size: 密度图大小
        patch_size: patch大小
        sigma: 高斯分布的标准差系数
    
    返回:
        torch.Tensor: 高斯标签张量，如果失败返回None
    """
    grid_shape = (cube_size[0] // patch_size,
                  cube_size[1] // patch_size,
                  cube_size[2] // patch_size)
    
    # 查找PDB文件
 
    # 读取密度图原点
    # resizemap = os.path.join(folder_path, "resize360.map")
    # map_origin = read_map_origin(resizemap)
    #print(f"使用密度图原点: {map_origin}")
    
    # 获取CA坐标
    scaled_pdb_path = os.path.join(folder_path, "scaled")
    if not os.path.exists(scaled_pdb_path):
        scaled_pdb_path = os.path.join(folder_path, "scaled.pdb")
    
    if not os.path.exists(scaled_pdb_path):
        print(f"找不到scaled PDB文件: {scaled_pdb_path}")
        return None
    
    scaledCA_Coordinates = get_ca_coords(scaled_pdb_path)
    if not scaledCA_Coordinates:
        print(f"未找到CA原子")
        return None
    
    #print(f"找到 {len(scaledCA_Coordinates)} 个CA原子")
    
    # 生成高斯标签
    gaussian_labels = []
    for i, CA in enumerate(scaledCA_Coordinates):
        # 计算patch ID
        patch_id = which_patch_pdb_ca(CA['coord'],grid_shape=grid_shape, patch_size=patch_size)
        
        # 生成高斯概率分布
        CAprobs = generate_gaussian_probability(patch_id, grid_shape, patch_size, sigma=sigma)
        gaussian_labels.append(CAprobs)
        
        # 调试信息（只打印前3个）
        # if i < 3:
            # print(f"  CA {i}: 坐标={CA['coord']}, patch_id={patch_id}, max_prob={np.max(CAprobs):.4f}")
    
    # 转换为tensor
    gaussian_labels_array = np.array(gaussian_labels)
    true_coords = torch.from_numpy(gaussian_labels_array).float()
    
    return true_coords

# 可选：添加验证函数
def verify_labels(true_coords, threshold=0.01):
    """验证生成的标签是否合理"""
    if true_coords is None:
        return False
    
    # 检查每个残基的概率分布
    for i, probs in enumerate(true_coords):
        # 确保概率和为1
        prob_sum = probs.sum().item()
        if abs(prob_sum - 1.0) > 0.001:
            print(f"警告：残基 {i} 的概率和为 {prob_sum}")
            return False
        
        # 检查是否有明显的峰值
        max_prob = probs.max().item()
        if max_prob < threshold:
            print(f"警告：残基 {i} 的最大概率过低 {max_prob}")
    
    return True

#true_coords = generate_gaussian_labels(scaled)

# if __name__ == "__main__":
    # pdb_file = '/ziyingz/Programs/E3-CryoFold/examples/scaled.pdb'
    # cube_size = (360, 360, 360)      # 按你的密度图实际设置
    # patch_size = 36
    # origin = (0, 0, 0)               # 如果有mrc origin，记得调整

    # cas = get_ca_coords(pdb_file)
    # for ca in cas:
        # ijk, patch_id = which_patch_pdb_ca(ca['coord'], patch_size, grid_shape, origin)
        # print(f"Chain {ca['chain']} Residue {ca['resid']} 坐标{ca['coord']} 在patch {ijk} (编号{patch_id})")



    # from collections import Counter
    # patch_ids = [which_patch_pdb_ca(ca['coord'], patch_size, grid_shape, origin)[1] for ca in cas]
    # counter = Counter(patch_ids)
    # print("每个patch中CA原子数量:", counter)

    # selected_patches = set(tuple(which_patch_pdb_ca(ca['coord'], patch_size, grid_shape, origin)[0]) for ca in cas)

    # import numpy as np
    # import mrcfile

# # # 1. 读入原始密度图（假设文件名为 '1.map'）
# # with mrcfile.open('/ziyingz/Programs/E3-CryoFold/resize360.map', permissive=True) as mrc:
    # # map_data = mrc.data.copy()  # 建议copy防止只读模式下出错

# # # 2. 选中patch
# # selected_map = np.zeros_like(map_data)
# # for i, j, k in selected_patches:
    # # z_start, z_end = i*patch_size, (i+1)*patch_size
    # # y_start, y_end = j*patch_size, (j+1)*patch_size
    # # x_start, x_end = k*patch_size, (k+1)*patch_size
    # # selected_map[z_start:z_end, y_start:y_end, x_start:x_end] = map_data[z_start:z_end, y_start:y_end, x_start:x_end]

# # # 3. 保存新密度图
# # with mrcfile.new('/ziyingz/Programs/E3-CryoFold/selected_patch.mrc', overwrite=True) as mrc:
    # # mrc.set_data(selected_map.astype(np.float32))


    # import numpy as np
    # import mrcfile

    # patch_size = 36  # 假设 patch 尺寸为 36
    # i, j, k = 6,5,4 # 只选中这一个块

    # # 1. 读入原始密度图
    # with mrcfile.open('/ziyingz/Programs/E3-CryoFold/resize360.map', permissive=True) as mrc:
        # map_data = mrc.data.copy()

    # # 2. 创建全零的 map，然后只复制你要的 patch
    # selected_map = np.zeros_like(map_data)
    # z_start, z_end = i*patch_size, (i+1)*patch_size
    # y_start, y_end = j*patch_size, (j+1)*patch_size
    # x_start, x_end = k*patch_size, (k+1)*patch_size
    # selected_map[z_start:z_end, y_start:y_end, x_start:x_end] = map_data[z_start:z_end, y_start:y_end, x_start:x_end]

    # # 3. 保存新密度图
    # with mrcfile.new('/ziyingz/Programs/E3-CryoFold/selected_patch654.mrc', overwrite=True) as mrc:
        # mrc.set_data(selected_map.astype(np.float32))


