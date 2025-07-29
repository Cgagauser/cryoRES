import torch
import numpy as np
import os
import json
from copy import deepcopy
import mrcfile
from scipy.ndimage import zoom
from Bio import PDB
from AAProNEWfeature import get_seq_blosum_physicochem_feat

alphabet = ['<cls>', '<pad>', '<eos>', '<unk>', 'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-', '<null_1>', '<mask>']

AA_map = {
  "ALA": "A",
  "CYS": "C",
  "ASP": "D",
  "GLU": "E",
  "PHE": "F",
  "GLY": "G",
  "HIS": "H",
  "ILE": "I",
  "LYS": "K",
  "LEU": "L",
  "MET": "M",
  "ASN": "N",
  "PRO": "P",
  "GLN": "Q",
  "ARG": "R",
  "SER": "S",
  "THR": "T",
  "VAL": "V",
  "TRP": "W",
  "TYR": "Y"
}
DNA_map = {
    'DA' : 'A',
    'DG' : 'G',
    'DC' : 'C',
    'DT' : 'T'
}
RNA_map = {
    'A' : 'A',
    'G' : 'G',
    'C' : 'C',
    'U' : 'U'
}

parser = PDB.PDBParser()

def resize_3d_data(data, target_shape):
    zoom_factors = (
        target_shape[0] / data.shape[0],
        target_shape[1] / data.shape[1],
        target_shape[2] / data.shape[2]
    )
    #print("zoom_factors:",zoom_factors)
    resized_data = zoom(data, zoom_factors, order=3) 
    zoom_factors = np.array(zoom_factors)
    return resized_data,zoom_factors



def save_resized_mrc(data, output_path):
    # 如果是 torch.Tensor，就先转成 numpy
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    
    data = data.astype(np.float32)  # 转换 dtype，确保兼容 MRC 写入
    with mrcfile.new(output_path, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.update_header_from_data()

def get_data(dir_path):
    protein_data, seq, chain_index = None, None, None

    for file in os.listdir(dir_path):
        if file.endswith('.mrc'):
            dm_path = os.path.join(dir_path, file)
            #print("processingmap:",dm_path)
            try:
                # p_map = mrcfile.open(dm_path, mode='r')
                # protein_data = deepcopy(p_map.data)
                # protein_data,zoom_factors = resize_3d_data(protein_data, [360, 360, 360])
                map_path = dir_path + "/resize360.map"
                with mrcfile.open(map_path, mode='r') as mrc:
                    protein_data = mrc.data  
                # if not os.path.exists(dir_path+"/resize360.map"):
                    # save_resized_mrc(protein_data,dir_path+"/resize360.map")
            except Exception as e:
                print(f"Error loading or processing the density map: {e}")
                raise
        elif file.endswith('.json'):
            seq_chain_path = os.path.join(dir_path, file)
            try:
                with open(seq_chain_path, 'r') as f:
                    seq_chain = json.load(f)
                    seq, chain_index = seq_chain['seq'], seq_chain['chain_index']
                   
                    physicochem_feat=get_seq_blosum_physicochem_feat(seq)
                    #print(type(physicochem_feat))  # 检查 physicoem_feat 的类型
                    seq = np.array([alphabet.index(item) for item in seq])
                    chain_index = np.array(chain_index)
            except Exception as e:
                print(f"Error loading or processing the seq/chain: {e}")
                raise
    
    if protein_data is None:
        raise FileNotFoundError("No .mrc file found in the specified directory or failed to load.")
    if seq is None or chain_index is None:
        raise FileNotFoundError("No valid .json file found in the specified directory or failed to load.")
    #return protein_data, seq, chain_index,physicochem_feat,zoom_factors
    return protein_data, seq, chain_index,physicochem_feat

def get_coord_from_pdb(pdb_path):
    if not os.path.exists(pdb_path):
        print(f"Error: PDB file '{pdb_path}' does not exist.")
        return None

    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("", pdb_path)
    except Exception as e:
        print(f"Error parsing PDB file '{pdb_path}': {e}")
        return None

    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                coord = np.zeros((4, 3), dtype=np.float32)                
                for i, atom in enumerate(residue):
                    if i >= 4:
                        break
                    coord[i] = atom.coord
                coords.append(coord)

    if not coords:
        print("Warning: No atomic coordinates were extracted.")
        return None
    
    return np.array(coords, dtype=np.float32)
