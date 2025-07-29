import sys
import os
from Bio.PDB import PDBParser

residue_dict = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',

    # 扩展氨基酸（或模糊符号）
    'ASX': 'B',  # ASP or ASN
    'GLX': 'Z',  # GLU or GLN
    'SEC': 'U',  # Selenocysteine
    'PYL': 'O',  # Pyrrolysine
    'XAA': 'X',  # Unknown

    # 可选：保留符号，映射为其自身或空字符（根据用途）
    '.': '.', 
    '-': '-', 
}

pdb_file = sys.argv[1]
output_file = sys.argv[2]

parser = PDBParser(QUIET=True)
structure = parser.get_structure('protein', pdb_file)

seq_combined = ""
chain_index = []

chains = list(structure.get_chains())

for idx, chain in enumerate(chains,start=1):
    residues = [residue for residue in chain if residue.get_id()[0] == ' ']
    for residue in residues:
        resname = residue.resname
        if resname in residue_dict:
            seq_combined += residue_dict[resname]
            chain_index.append(idx)
        else:
            print(f"⚠️ 未知残基: {resname}，使用 X 占位")
            seq_combined += 'X'
            chain_index.append(idx)
# print(len(seq_combined))
# print(len(chain_index))
assert len(seq_combined) == len(chain_index), "序列与链编号长度不匹配！"

with open(output_file, "w") as f:
    f.write('{\n')
    f.write(f'    "seq": "{seq_combined}",\n')
    f.write('    "chain_index": [')
    f.write(', '.join(map(str, chain_index)))
    f.write(']\n')
    f.write('}\n')

print(f"JSON数据已保存到{output_file}")

 
 
 #1000 atoms to a JSON file
# def write_json_file(seq, chain_idx, output_path, chunk_num):
    # """写入JSON文件"""
    # with open(f"{output_path}_{chunk_num}.json", "w") as f:
        # f.write('{\n')
        # f.write(f'    "seq": "{seq}",\n')
        # f.write('    "chain_index": [')
        # f.write(', '.join(map(str, chain_idx)))
        # f.write(']\n')
        # f.write('}\n')

# def split_sequence(seq, chain_idx, max_length=1000):
    # """将序列分割成指定长度的片段"""
    # chunks = []
    # chain_chunks = []
    # for i in range(0, len(seq), max_length):
        # chunks.append(seq[i:i + max_length])
        # chain_chunks.append(chain_idx[i:i + max_length])
    # return chunks, chain_chunks

# # ...existing code...
# residue_dict = {
    # 'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    # 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    # 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    # 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
# }

# pdb_file = sys.argv[1]
# output_file = sys.argv[2]

# parser = PDBParser(QUIET=True)
# structure = parser.get_structure('protein', pdb_file)

# seq_combined = ""
# chain_index = []

# chains = list(structure.get_chains())

# for idx, chain in enumerate(chains,start=1):
    # residues = [residue for residue in chain if residue.get_id()[0] == ' ']
    # for residue in residues:
        # resname = residue.resname
        # if resname in residue_dict:
            # seq_combined += residue_dict[resname]
            # chain_index.append(idx)
        # else:
            # print(f"跳过非标准残基: {resname}")

# print(f"总序列长度: {len(seq_combined)}")
# print(f"总链索引长度: {len(chain_index)}")
# assert len(seq_combined) == len(chain_index), "序列与链编号长度不匹配！"

# # 分割序列和链索引
# output_path = os.path.splitext(output_file)[0]
# seq_chunks, chain_chunks = split_sequence(seq_combined, chain_index)

# # 写入多个JSON文件
# for i, (seq_chunk, chain_chunk) in enumerate(zip(seq_chunks, chain_chunks)):
    # write_json_file(seq_chunk, chain_chunk, output_path, i+1)
    # print(f"已保存第{i+1}个文件，长度为{len(seq_chunk)}")

# print(f"共生成了{len(seq_chunks)}个JSON文件")



#a chain to a JSON files
# import os
# import sys
# from Bio.PDB import PDBParser

# def write_json_file(seq, chain_idx, output_path, chain_id):
    # """写入JSON文件"""
    # with open(f"{output_path}_chain_{chain_id}.json", "w") as f:
        # f.write('{\n')
        # f.write(f'    "seq": "{seq}",\n')
        # f.write('    "chain_index": [')
        # f.write(', '.join(map(str, chain_idx)))
        # f.write(']\n')
        # f.write('}\n')

# # 三字母氨基酸转一字母
# residue_dict = {
    # 'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    # 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    # 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    # 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
# }

# # 读取 PDB 文件路径和输出路径
# pdb_file = sys.argv[1]
# output_file = sys.argv[2]  # 示例: output.json

# parser = PDBParser(QUIET=True)
# structure = parser.get_structure('protein', pdb_file)

# output_path = os.path.splitext(output_file)[0]  # 去掉 .json

# chains = list(structure.get_chains())

# for idx, chain in enumerate(chains, start=1):
    # seq = ""
    # chain_idx = []
    # residues = [res for res in chain if res.get_id()[0] == ' ']
    # for residue in residues:
        # resname = residue.resname
        # if resname in residue_dict:
            # seq += residue_dict[resname]
            # chain_idx.append(idx)
        # else:
            # print(f"跳过非标准残基: {resname}")

    # if seq:
        # write_json_file(seq, chain_idx, output_path, chain.id)
        # print(f"已保存链 {chain.id} 的 JSON 文件，长度为 {len(seq)}")
    # else:
        # print(f"链 {chain.id} 没有有效残基，跳过。")

# print(f"共处理了 {len(chains)} 条链")
