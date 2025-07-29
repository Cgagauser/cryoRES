#!/usr/bin/env python3
"""
批量检查子文件夹中的JSON和scaled文件一致性
删除不一致的文件夹
"""

import os
import json
import sys
import numpy as np
import shutil
from pathlib import Path

def get_ca_coords(pdb_path):
    """从PDB文件获取CA坐标,正确处理多构象"""
    ca_coords = []
    seen_residues = set()  # 记录已经处理的残基
    
    try:
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                    resid = int(line[22:26])
                    chain = line[21]
                    altloc = line[16]  # 备选位置指示符(第17列)
                    
                    # 构建唯一的残基标识
                    residue_key = (chain, resid)
                    
                    # 如果这个残基已经处理过,跳过(只保留第一个构象)
                    if residue_key in seen_residues:
                        continue
                    
                    # 如果有备选位置,只取第一个(通常是'A'或空格)
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
        return None
    
    return ca_coords

def check_consistency(json_file, pdb_file):
    """检查JSON序列长度和PDB残基数是否一致"""
    
    # 读取JSON文件
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        seq = data.get('seq', '')
        seq_length = len(seq)
    except Exception as e:
        print(f"    ❌ 读取JSON文件失败: {e}")
        return False
    
    # 获取PDB残基数
    ca_coords = get_ca_coords(pdb_file)
    if ca_coords is None:
        print(f"    ❌ 读取PDB文件失败")
        return False
    
    pdb_residue_count = len(ca_coords)
    
    # 比较结果
    if seq_length == pdb_residue_count:
        print(f"    ✅ 一致: JSON({seq_length}) = PDB({pdb_residue_count})")
        return True
    else:
        print(f"    ❌ 不一致: JSON({seq_length}) ≠ PDB({pdb_residue_count}), 差异: {abs(seq_length - pdb_residue_count)}")
        return False

def process_directory(root_dir, dry_run=False):
    """处理目录下的所有子文件夹"""
    
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"错误: 目录 {root_dir} 不存在")
        return
    
    folders_to_delete = []
    total_folders = 0
    consistent_folders = 0
    
    print(f"开始扫描目录: {root_dir}")
    print("=" * 60)
    
    # 遍历所有子文件夹
    for subdir in root_path.iterdir():
        if not subdir.is_dir():
            continue
            
        total_folders += 1
        print(f"\n检查文件夹: {subdir.name}")
        
        # 查找JSON和scaled文件
        json_files = list(subdir.glob("*.json"))
        scaled_files = list(subdir.glob("scaled")) + list(subdir.glob("*scaled*.cif"))
        
        if not json_files:
            print("    ⚠️  未找到JSON文件")
            continue
            
        if not scaled_files:
            print("    ⚠️  未找到scaled文件")
            continue
        
        # 使用找到的第一个JSON和scaled文件
        json_file = json_files[0]
        scaled_file = scaled_files[0]
        
        print(f"    JSON文件: {json_file.name}")
        print(f"    Scaled文件: {scaled_file.name}")
        
        # 检查一致性
        is_consistent = check_consistency(json_file, scaled_file)
        
        if is_consistent:
            consistent_folders += 1
        else:
            folders_to_delete.append(subdir)
    
    # 统计结果
    print("\n" + "=" * 60)
    print(f"扫描完成！")
    print(f"总文件夹数: {total_folders}")
    print(f"一致的文件夹: {consistent_folders}")
    print(f"不一致的文件夹: {len(folders_to_delete)}")
    
    # 删除不一致的文件夹
    if folders_to_delete:
        print(f"\n将要删除的文件夹 ({len(folders_to_delete)} 个):")
        for folder in folders_to_delete:
            print(f"  - {folder.name}")
        
        if dry_run:
            print("\n⚠️  DRY RUN模式: 不会真正删除文件夹")
        else:
            print("\n⚠️  警告: 即将删除上述文件夹！")
            confirm = input("确认删除? (yes/no): ")
            
            if confirm.lower() == 'yes':
                for folder in folders_to_delete:
                    try:
                        shutil.rmtree(folder)
                        print(f"  ✅ 已删除: {folder.name}")
                    except Exception as e:
                        print(f"  ❌ 删除失败 {folder.name}: {e}")
            else:
                print("取消删除操作")
    else:
        print("\n✅ 所有文件夹都一致，无需删除")

def main():
    if len(sys.argv) < 2:
        print("用法: python batch_check.py <目录路径> [--dry-run]")
        print("  --dry-run: 只检查，不删除文件夹")
        sys.exit(1)
    
    root_dir = sys.argv[1]
    dry_run = "--dry-run" in sys.argv
    
    process_directory(root_dir, dry_run)

if __name__ == "__main__":
    main()