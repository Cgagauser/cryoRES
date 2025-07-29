import numpy as np
from Bio.PDB import PDBParser


def get_atom_voxel_coords(pdb_file, origin, voxel_size, scale_factor=1.8):
    """
    将 PDB 中原子坐标转换为 voxel 坐标，并将坐标扩大 scale_factor 倍（与 360³ 密度图对齐）
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('pdb', pdb_file)
    coords = []

    for atom in structure.get_atoms():
        pos = atom.get_coord()  # 单位为 Å
        scaled_pos = pos * scale_factor  # 扩大1.8倍
        #rint(f"Original pos: {pos}, Scaled pos: {scaled_pos}")  # 打印原始和扩大的坐标
        voxel = ((scaled_pos - origin) / voxel_size).astype(int)  # 转成 voxel 坐标
        coords.append(voxel)
        print(voxel)
    return np.array(coords)  # shape: [N_atoms, 3]

def get_atom_voxel_coords(pdb_file, origin, voxel_size):
    """
    将 PDB 中原子坐标转换为 voxel 坐标，并将坐标扩大 scale_factor 倍（与 360³ 密度图对齐）
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('pdb', pdb_file)
    coords = []

    for atom in structure.get_atoms():
        pos = atom.get_coord()  # 单位为 Å
        #rint(f"Original pos: {pos}, Scaled pos: {scaled_pos}")  # 打印原始和扩大的坐标
        voxel = ((pos - origin) / voxel_size).astype(int)  # 转成 voxel 坐标
        coords.append(voxel)
        print(voxel)
    return np.array(coords)  # shape: [N_atoms, 3]

def build_patch_labels_from_coords(voxel_coords, patch_size=36, grid_shape=(10,10,10)):
    labels = np.zeros(np.prod(grid_shape), dtype=np.int32)

    for x, y, z in voxel_coords:
        i, j, k = x // patch_size, y // patch_size, z // patch_size
        if 0 <= i < grid_shape[0] and 0 <= j < grid_shape[1] and 0 <= k < grid_shape[2]:
            idx = i * (grid_shape[1] * grid_shape[2]) + j * grid_shape[2] + k  # Flatten index
            labels[idx] = 1

    return labels  # shape: [1000], 每个 patch 是否含有原子

#可视化label
import numpy as np
import mrcfile

def patch_labels_to_voxel_mask(labels, patch_size=36, grid_shape=(10,10,10), volume_shape=(360,360,360)):
    """
    将 patch 标签 [1000] 映射为 3D 掩膜 [360,360,360]
    """
    mask = np.zeros(volume_shape, dtype=np.float32)

    for idx, val in enumerate(labels):
        if val == 0:
            continue
        i = idx // (grid_shape[1] * grid_shape[2])
        j = (idx // grid_shape[2]) % grid_shape[1]
        k = idx % grid_shape[2]

        x0, y0, z0 = i * patch_size, j * patch_size, k * patch_size
        x1, y1, z1 = x0 + patch_size, y0 + patch_size, z0 + patch_size

        mask[x0:x1, y0:y1, z0:z1] = 1.0

    return mask  # shape: [360,360,360]

def save_mask_as_mrc(mask, output_path, voxel_size=1.0, origin=(0,0,0)):
    """
    保存 mask 为 .mrc 文件
    """
    with mrcfile.new(output_path, overwrite=True) as mrc:
        mrc.set_data(mask.astype(np.float32))
        mrc.voxel_size = voxel_size
        mrc.header.origin = origin
        mrc.update_header_from_data()
        mrc.update_header_stats()



pdb_file = "/ziyingz/Programs/E3-CryoFold/traindata/5uz7all/5uz7allscaled_output.pdb"
origin = np.array([0.0, 0.0, 0.0])       # 原点坐标（密度图中 origin）
voxel_size = np.array([1.0, 1.0, 1.0])   # spacing = 1 Å, 因为你已经 resize 到 360³
coords = get_atom_voxel_coords(pdb_file, origin, voxel_size)
#coords = get_atom_voxel_coords(pdb_file, origin, voxel_size, scale_factor=1.8)

# 确保grid_shape正确（根据你的密度图分辨率）
grid_shape = (360, 360, 360)  # 这应该与你的密度图大小一致
labels = build_patch_labels_from_coords(coords, patch_size=36, grid_shape=grid_shape)


# 已有 labels: shape = [1000]
voxel_mask = patch_labels_to_voxel_mask(labels)  # → shape [360, 360, 360]

save_mask_as_mrc(voxel_mask, "/ziyingz/Programs/E3-CryoFold/examples/density_map/structure_masknew.mrc", voxel_size=1.0, origin=(0, 0, 0))
'''from Bio.PDB import PDBIO

def generate_scaled_pdb(pdb_file, output_pdb, scale_factor=1.8):
    """
    生成放大坐标的PDB文件
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('pdb', pdb_file)

    # 创建 PDB 输出对象
    io = PDBIO()

    # 遍历结构中的每个原子，并放大坐标
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    # 放大原子坐标
                    atom.coord = atom.coord * scale_factor
                    #print(f"Scaled position for atom {atom.get_name()}: {atom.coord}")

    # 将放大后的结构保存到新PDB文件
    io.set_structure(structure)
    io.save(output_pdb)  # 保存为新的PDB文件

# 使用示例
input_pdb_file = "/ziyingz/Programs/E3-CryoFold/traindata/5uz7all/5uz7all.pdb"
output_pdb_file = "/ziyingz/Programs/E3-CryoFold/traindata/5uz7all/5uz7allscaled_output.pdb"
generate_scaled_pdb(input_pdb_file, output_pdb_file, scale_factor=1.8)'''