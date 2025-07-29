import os
import argparse
import torch
from src.chroma.data import Protein
from model import CryoFold
from utils import align, get_data, get_coord_from_pdb
import argparse
from tqdm import tqdm
from datetime import datetime
import sys 
from label import *
import numpy as np



def parse_arguments():
    parser = argparse.ArgumentParser(description="CryoFold: density map to protein structure")
    parser.add_argument('--density_map_path', type=str,default="/ziyingz/Programs/E3-CryoFold/testtemp/7XXF_4_H", help='Path to the input data directory')
    parser.add_argument('--pdb_path', type=str, default=None, help='Path to the ground truth PDB file')
    parser.add_argument('--model_path', type=str, default='/ziyingz/Programs/E3-CryoFold/checkpoint/checkpoint_epoch_9.pt', 
                        help='Path to the pretrained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='results', 
                        help='Directory to save the output PDB file')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda', 
                        help='Device to run the model on')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output for debugging')
    return parser.parse_args()


def load_model(model_path: str, device: str) -> CryoFold:
    cryofold = CryoFold(
        img_shape=(360, 360, 360), input_dim=1, output_dim=4, embed_dim=512,
        patch_size=36, num_heads=8, dropout=0.1, ext_layers=[3, 6, 9, 12], 
        norm="instance", decoder_dim=128
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    try:
        cryofold.load_state_dict(state_dict)
    except:
        pass
        # state_dict = {k.replace('_forward_module.model.',''):v for k,v in state_dict.items()}
        # cryofold.load_state_dict(state_dict,strict=False)
    return cryofold


def preprocess_data(dir_path: str, device: str = 'cuda'):
    print('Preprocessing data...')
    data = get_data(dir_path)
    maps, seq, chain_encoding,physicochem_feat,zoom_factors = (torch.from_numpy(x).to(device) for x in data)
    pdb_files = [f for f in os.listdir(dir_path) if f.endswith('.pdb')]
    # if pdb_files:
        # pdb_path = os.path.join(dir_path, pdb_files[0])
        # if not os.path.exists(dir_path+"/scaled"):
            # with open(pdb_path) as fin, open(dir_path+"/scaled", "w") as fout:
                # for line in fin:
                    # if line.startswith("ATOM"):
                        # x = float(line[30:38]) * zoom_factors[2]
                        # y = float(line[38:46]) * zoom_factors[1]
                        # z = float(line[46:54]) * zoom_factors[0]
                        # fout.write(f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}")
                    # else:
                        # fout.write(line)
        # scaledCA_Coordinates=get_ca_coords(dir_path+"/scaled")
        # gaussian_labels = []
        # for CA in scaledCA_Coordinates:
            # patch_id = which_patch_pdb_ca(CA['coord'])
            # print(patch_id)
    CAprobs=generate_gaussian_labels(dir_path)
        # gaussian_labels.append(CAprobs)
    gaussian_labels_array = np.array(CAprobs)  # 先转换为numpy数组
    true_coords = torch.from_numpy(gaussian_labels_array).float()  # 再转换为tensor
    max_patch_indices = torch.argmax(true_coords, dim=1)  # [L]，每个残基概率最大的patch索引
    max_probs = torch.max(true_coords, dim=1).values
    for i, (idx, prob) in enumerate(zip(max_patch_indices.tolist(), max_probs.tolist())):
        print(f"Residue {i}: patch {idx}, max prob {prob:.5f}")
    print('Preprocessing finished!')
    return maps, seq, chain_encoding,physicochem_feat,zoom_factors


'''def infer_structure(model: CryoFold, maps: torch.Tensor, seq: torch.Tensor, 
                    chain_encoding: torch.Tensor, coords: torch.Tensor = None):
    try:
        pred_x, _ = model.infer(maps, seq, chain_encoding)
        if coords is not None:
            pred_x, rmsd = align(pred_x, coords)
            print(f'RMSD: {rmsd:.4f}')
    except Exception as e:
        print(f"Error during inference: {e}")
    return pred_x, seq, chain_encoding'''

def infer_structure(model: CryoFold, maps: torch.Tensor, seq: torch.Tensor, 
                    chain_encoding: torch.Tensor, coords: torch.Tensor, physicochem_feat: torch.Tensor, zoom_factors):

        pred_x = model.infer(maps, seq, chain_encoding,physicochem_feat)
        return pred_x


'''def save_protein(preds: torch.Tensor, seqs: torch.Tensor, chain_encodings: torch.Tensor, output_dir: str, output_name: str = 'output'):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name + '.pdb')

    # Process sequence values to ensure they fall within a valid range
    seqs[seqs > 19] = 0
    protein = Protein.from_XCS(preds.unsqueeze(0), chain_encodings.unsqueeze(0), seqs.unsqueeze(0))
    protein.to_PDB(output_path)
    print(f"Protein structure saved to {output_path}")'''


def save_pdb(preds: torch.Tensor, seqs: torch.Tensor, chain_encodings: torch.Tensor, output_dir: str, output_name: str = 'output'):
    # 添加氨基酸映射字典
    aa_dict = {
        4: 'LEU', 5: 'ALA', 6: 'GLY', 7: 'VAL', 8: 'SER',
        9: 'GLU', 10: 'ARG', 11: 'THR', 12: 'ILE', 13: 'ASP',
        14: 'PRO', 15: 'LYS', 16: 'GLN', 17: 'ASN', 18: 'PHE',
        19: 'TYR', 20: 'MET', 21: 'HIS', 22: 'TRP', 23: 'CYS',
        24: 'UNK',  # X -> UNK
        25: 'ASX',  # B -> ASX (ASP/ASN)
        26: 'SEC',  # U -> SEC (硒代半胱氨酸)
        27: 'GLX',  # Z -> GLX (Glu/Gln)
        28: 'PYL'   # O -> PYL (吡咯赖氨酸)
    }
    #print(seqs)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name + '.pdb')
    
    atom_types = ['N', 'CA', 'C', 'O']
    
    with open(output_path, 'w') as f:
        f.write("REMARK   Generated by CryoFold\n")
        
        atom_index = 1
        for res_idx in range(preds.shape[0]):
            res_num = res_idx + 1
            seq_num = seqs[res_idx].item()
            aa_type = aa_dict.get(seq_num, 'UNK')  # 如果找不到映射，使用UNK
            chain_id = chr(64 + chain_encodings[res_idx].item())
            
            for atom_idx, atom_type in enumerate(atom_types):
                coords = preds[res_idx, atom_idx]
                x, y, z = coords.tolist()
                
                line = f"ATOM  {atom_index:5d}  {atom_type:<3s} {aa_type:>3s} {chain_id}{res_num:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_type[0]}\n"
                f.write(line)
                atom_index += 1
        
        f.write("END\n")   
    print(f"Protein structure saved to {output_path}")



def main():
    args = parse_arguments()
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    
    if args.verbose:
        print(f"Running on device: {device}")

    model = load_model(args.model_path, device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("模型可训练参数总数:", total_params)

    """     total_params = sum(p.numel() for p in model.parameters())
    print("模型全部参数数:", total_params) """

    """     for name, param in model.named_parameters():
        print(f"{name}: {param.numel()}") """


    maps, seq, chain_encoding,physicochem_feat,zoom_factors = preprocess_data(args.density_map_path, device)
    print("physicochem_feat.shape:",physicochem_feat.shape)
    print("zoom_factors:",zoom_factors)

    print("seqshape:",seq.shape)

    coords = None
    if args.pdb_path:
        coords = get_coord_from_pdb(args.pdb_path)
        coords = torch.from_numpy(coords).to(device)
    Xpred=infer_structure(model, maps, seq, chain_encoding, coords,physicochem_feat,zoom_factors)
    Xpred = Xpred.squeeze(0) 
    max_probs, pred_idx = Xpred.max(dim=-1) 
    for i, (idx, prob) in enumerate(zip(pred_idx.tolist(), max_probs.tolist())):
        print(f"Residue {i}: predicted patch {idx}, max prob {prob:.5f}")
if __name__ == "__main__":
    main()
    
    
    
    
    
    
# import os
# import argparse
# import torch
# from src.chroma.data import Protein
# from model import CryoFold
# from utils import align, get_data, get_coord_from_pdb, alphabet


# def parse_arguments():
    # parser = argparse.ArgumentParser(description="CryoFold: density map to protein structure")
    # parser.add_argument('--density_map_path', type=str, default=None, help='Path to the input data directory')
    # parser.add_argument('--pdb_path', type=str, default=None, help='Path to the ground truth PDB file')
    # parser.add_argument('--model_path', type=str, default='/ziyingz/Programs/E3-CryoFold/pretrained_model/checkpoint.pt', 
                        # help='Path to the pretrained model checkpoint')
    # parser.add_argument('--output_dir', type=str, default='results', 
                        # help='Directory to save the output PDB file')
    # parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda', 
                        # help='Device to run the model on')
    # parser.add_argument('--verbose', action='store_true', help='Enable verbose output for debugging')
    # return parser.parse_args()


# # def load_model(model_path: str, device: str) -> CryoFold:
    # # cryofold = CryoFold(
        # # img_shape=(360, 360, 360), input_dim=1, output_dim=4, embed_dim=480,
        # # patch_size=36, num_heads=8, dropout=0.1, ext_layers=[3, 6, 9, 12], 
        # # norm="instance", decoder_dim=128
    # # ).to(device)
    # # checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    # # try:
        # # cryofold.load_state_dict(checkpoint)
    # # except:
        # # checkpoint = {k.replace('_forward_module.model.',''):v for k,v in checkpoint.items() if 'loss_' not in k}
        # # cryofold.load_state_dict(checkpoint)
    # # return cryofold

# # def load_model(model_path: str, device: str) -> CryoFold:
    # # cryofold = CryoFold(
        # # img_shape=(360, 360, 360), input_dim=1, output_dim=4, embed_dim=480,
        # # patch_size=36, num_heads=8, dropout=0.1, ext_layers=[3, 6, 9, 12], 
        # # norm="instance", decoder_dim=128
    # # ).to(device)
    # # checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    # # try:
        # # cryofold.load_state_dict(checkpoint)
    # # except:
        # # checkpoint = {k.replace('_forward_module.model.',''):v for k,v in checkpoint.items()}
        # # cryofold.load_state_dict(checkpoint,strict=False)
    # # return cryofold





# # def preprocess_data(dir_path: str, device: str = 'cuda'):
    # # print('Preprocessing data...')
    # # data = get_data(dir_path)
    # # maps, seq, chain_encoding = (torch.from_numpy(x).to(device) for x in data)
    # # print('Preprocessing finished!')
    # # return maps, seq, chain_encoding


# # def infer_structure(model: CryoFold, maps: torch.Tensor, seq: torch.Tensor, 
                    # # chain_encoding: torch.Tensor, coords: torch.Tensor = None):
    # # try:
        # # pred_x, _ = model.infer(maps, seq, chain_encoding)
        # # if coords is not None:
            # # pred_x, rmsd = align(pred_x, coords)
            # # print(f'RMSD: {rmsd:.4f}')
    # # except Exception as e:
        # # print(f"Error during inference: {e}")
    # # return pred_x, seq, chain_encoding


# # def save_protein(preds: torch.Tensor, seqs: torch.Tensor, chain_encodings: torch.Tensor, output_dir: str, output_name: str = 'output'):
    # # os.makedirs(output_dir, exist_ok=True)
    # # output_path = os.path.join(output_dir, output_name + '.pdb')

    # Process sequence values to ensure they fall within a valid range
    # # protein = Protein.from_XCS(preds.unsqueeze(0), chain_encodings.unsqueeze(0), seqs.unsqueeze(0), alphabet=alphabet)
    # # protein.to_PDB(output_path)
    # # print(f"Protein structure saved to {output_path}")


# # def main():
    # # args = parse_arguments()
    # # device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    
    # # if args.verbose:
        # # print(f"Running on device: {device}")

    # # model = load_model(args.model_path, device)
    # # maps, seq, chain_encoding = preprocess_data(args.density_map_path, device)

    # # coords = None
    # # if args.pdb_path:
        # # coords = get_coord_from_pdb(args.pdb_path)
        # # coords = torch.from_numpy(coords).to(device)

    # # preds, seqs, chain_encodings = infer_structure(model, maps, seq, chain_encoding, coords)
    # # save_protein(preds, seqs, chain_encodings, args.output_dir)

# # if __name__ == "__main__":
    # # main()