import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from model import CryoFold
from utils import get_data, align
import argparse
from tqdm import tqdm
from datetime import datetime
import sys 
import torch.nn.functional as F
from label import  generate_gaussian_labels, verify_labels
def parse_args():
    parser = argparse.ArgumentParser(description='E3-CryoFold Training')
    
    #数据相关参数
    parser.add_argument('--train_data_dir', type=str, default="/ziyingz/Programs/E3-CryoFold/testtemp", help='训练数据目录')
    parser.add_argument('--val_data_dir', type=str, default="/ziyingz/Programs/E3-CryoFold/testtemp", help='验证数据目录')
    parser.add_argument('--max_len', type=int, default=2500, help='最大序列长度')
    #模型相关参数
    parser.add_argument('--embed_dim', type=int, default=512, help='嵌入维度')
    parser.add_argument('--patch_size', type=int, default=36, help='密度图patch大小')
    parser.add_argument('--num_heads', type=int, default=2, help='注意力头数')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout率')
    parser.add_argument('--decoder_dim', type=int, default=128, help='解码器维度')
    #训练相关参数
    parser.add_argument('--batch_size', type=int, default=1, help='批次大小')
    parser.add_argument('--epochs', type=int, default=1000, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--max_grad_norm', type=float, default=0.1, help='梯度裁剪阈值')
    #其他参数
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='训练设备')
    parser.add_argument('--output_dir', type=str, default='/ziyingz/Programs/E3-CryoFold/checkpoint', help='模型保存目录')
    parser.add_argument('--save_every', type=int, default=100, help='每多少个epoch保存一次模型')
    parser.add_argument('--pretrained_model', type=str, default=None,help='预训练模型路径')

    return parser.parse_args()

class CryoFoldDataset(Dataset):
    def __init__(self, data_dir, device='cpu'):  
        self.data_dir = data_dir
        self.device = device  
        self.data_folders = [os.path.join(data_dir, d) for d in os.listdir(data_dir) 
                            if os.path.isdir(os.path.join(data_dir, d))]    
    def __len__(self):
        return len(self.data_folders)
    def __getitem__(self, idx):
        folder_path = self.data_folders[idx]
        print("Processing:", os.path.basename(folder_path))
        data = get_data(folder_path)
        maps, seq, chain_encoding, physicochem_feat = (torch.from_numpy(x.copy()) for x in data) 
        true_coords = generate_gaussian_labels(folder_path)
        length = true_coords.shape[0]
        first_pos_true = torch.argmax(true_coords[0, :])
        mid_pos_true = torch.argmax(true_coords[length//2, :]) if length > 1 else first_pos_true
        last_pos_true = torch.argmax(true_coords[-1, :]) 
        true_info = {
            'first_pos': first_pos_true.item(),
            'mid_pos': mid_pos_true.item(),
            'last_pos': last_pos_true.item()
        }
        return {
            'maps': maps,
            'seq': seq,
            'chain_encoding': chain_encoding,
            'physicochem_feat': physicochem_feat,
            'true_coords': true_coords,
            'true_info': true_info,
            'folder_name': os.path.basename(folder_path)
        }

# def criterion(patch_prob, true_coords):
    # true_labels = torch.argmax(true_coords, dim=-1)
    # criterion = nn.CrossEntropyLoss()
    # loss = criterion(patch_prob.view(-1, patch_prob.size(-1)), 
                    # true_labels.view(-1))
    # return loss
       
def train(args):
    # 设置设备
    log_path = os.path.join( "/ziyingz/Programs/E3-CryoFold/train.csv")
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write("epoch,train_loss,val_loss\n")

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    # 创建数据集和数据加载器
    train_dataset = CryoFoldDataset(args.train_data_dir,args.device)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    if args.val_data_dir:
        val_dataset = CryoFoldDataset(args.val_data_dir,args.device)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    # 创建模型
    model = CryoFold(
        img_shape=(360, 360, 360), 
        input_dim=1, 
        output_dim=4, 
        embed_dim=args.embed_dim,
        patch_size=args.patch_size, 
        num_heads=args.num_heads, 
        dropout=args.dropout, 
        ext_layers=[3, 6, 9, 12], 
        norm="instance", 
        decoder_dim=args.decoder_dim
    ).to(device)
    
   
    # 1. 冻结ESM预训练模型
    esm_frozen = 0
    for name, param in model.named_parameters():
        if 'esm' in name.lower():
            param.requires_grad = False
            esm_frozen += 1

    # 2. 确保decoder可训练
    decoder_enabled = 0  
    for name, param in model.named_parameters():
        if ('decoder' in name.lower() or 
            'out' in name.lower() or 
            'classifier' in name.lower() or
            'patch_classifier' in name.lower()):
            param.requires_grad = True
            decoder_enabled += 1
            
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if module.weight.std() < 1e-6:
                nn.init.kaiming_normal_(module.weight)
                print(f"重新初始化: {module}")
    
    # 3. 替换problematic激活函数
    def replace_relu(module):
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                setattr(module, name, nn.GELU())
            else:
                replace_relu(child)
    
    replace_relu(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print("模型可训练参数总数:", total_params) 
    
    #optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        eps=1e-8, 
        weight_decay=0.01
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.KLDivLoss(reduction='batchmean')
    start_epoch = 0
    best_val_loss = float('inf')
    if args.pretrained_model:
        checkpoint = torch.load(args.pretrained_model, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(f"✅ 成功恢复模型、优化器、学习率调度器状态，从第 {start_epoch} 轮继续训练")
        else:
            model.load_state_dict(checkpoint)  # fallback 
            print("⚠️ 加载的是纯模型参数，优化器与学习率调度器未恢复")
   
    os.makedirs(args.output_dir, exist_ok=True)

    
    for epoch in range(start_epoch,args.epochs):

        log_file = os.path.join(args.output_dir, f"epoch_{epoch+1:03d}.log")
        with open(log_file, "w") as log_f:
            model.train()
            train_loss = 0.0
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")        
            for batch_idx, batch in enumerate(train_pbar):
                maps = batch['maps'].to(device)
                seq = batch['seq'].to(device)
                chain_encoding = batch['chain_encoding'].to(device)
                physicochem_feat = batch['physicochem_feat'].to(device)
                true_coords = batch['true_coords'].to(device) 
                true_info = batch['true_info']   
                optimizer.zero_grad()
                seq_pos = torch.arange(seq.shape[1], device=device).unsqueeze(0).repeat(seq.shape[0], 1)
                mask = torch.ones_like(seq).bool()      
                patch_prob, pred_info = model(maps, seq, seq_pos, chain_encoding,physicochem_feat, mask)
                folder_name = batch.get('folder_name', 'unknown')
                true_positions = f"第1个位置真实: {true_info['first_pos'][0]}, 中间位置真实: {true_info['mid_pos'][0]}, 最后位置真实: {true_info['last_pos'][0]}"
                pred_positions = f"第1个位置预测: {pred_info['first_pos']}, 中间位置预测: {pred_info['mid_pos']}, 最后位置预测: {pred_info['last_pos']}"
                print(true_positions)
                print(pred_positions)
                log_f.write(f"[Train] Batch {batch_idx+1} | Folder: {folder_name} | {true_positions}\n")
                log_f.write(f"[Train] Batch {batch_idx+1} | Folder: {folder_name} | {pred_positions}\n")
                #print(f"Processing Folder: {folder_name}")
                # 计算损失
                #org1#loss = criterion(patch_prob.view(-1, 1000), true_coords.view(-1))
                loss = criterion(patch_prob, true_coords)
                # is_collapsed = check_diversity(patch_prob)
                #loss = criterion(patch_prob, true_patch_prob)  
                # 反向传播和优化
                loss.backward()
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        print(f"[Grad] {name:50s} | norm = {grad_norm:.6f}")
                        
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step() 
                train_loss += loss.item()
                train_pbar.set_postfix({'runningLoss': train_loss / (batch_idx + 1)})
                print(f"[Train] Batch {batch_idx+1}/{len(train_loader)} | Folder: {folder_name} | batchLoss: {loss.item():.6f}")
                log_f.write(f"[Train] Batch {batch_idx+1}/{len(train_loader)} | Folder: {folder_name} | batchLoss: {loss.item():.6f}\n")
                log_f.flush()  # 保险起见及时写盘
                #torch.cuda.empty_cache()
            train_loss /= len(train_loader)
            print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.6f}")
            log_f.write(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.6f}\n")
            # 验证
            if args.val_data_dir:
                model.eval()
                val_loss = 0.0   
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
                with torch.no_grad():
                    for batch_idx, batch in enumerate(val_pbar):
                        maps = batch['maps'].to(device)
                        seq = batch['seq'].to(device)
                        chain_encoding = batch['chain_encoding'].to(device)
                        physicochem_feat = batch['physicochem_feat'].to(device)
                        true_coords = batch['true_coords'].to(device)
                        true_info = batch['true_info']
                       
                        seq_pos = torch.arange(seq.shape[1], device=device).unsqueeze(0).repeat(seq.shape[0], 1)
                        mask = torch.ones_like(seq).bool() 
                        patch_prob, pred_info = model(maps, seq, seq_pos, chain_encoding,physicochem_feat, mask)  
                        folder_name = batch.get('folder_name', 'unknown')
                        true_positions = f"第1个位置真实: {true_info['first_pos'][0]}, 中间位置真实: {true_info['mid_pos'][0]}, 最后位置真实: {true_info['last_pos'][0]}"
                        pred_positions = f"第1个位置预测: {pred_info['first_pos']}, 中间位置预测: {pred_info['mid_pos']}, 最后位置预测: {pred_info['last_pos']}"
                        print(f"[Val] {true_positions}")
                        print(f"[Val] {pred_positions}")
                        log_f.write(f"[Val] Batch {batch_idx+1} | Folder: {folder_name} | {true_positions}\n")
                        log_f.write(f"[Val] Batch {batch_idx+1} | Folder: {folder_name} | {pred_positions}\n")
                        #print(f"Processing Folder: {folder_name}")
                        #or1#loss = criterion(patch_prob.view(-1, 1000), true_coords.view(-1))
                        loss = criterion(patch_prob, true_coords)
                        val_loss += loss.item()
                        val_pbar.set_postfix({'runningLoss': val_loss / (batch_idx + 1)})
                        
                        print(f"[Val] Batch {batch_idx+1}/{len(val_loader)} | Folder: {folder_name} | batchLoss: {loss.item():.6f}")
                        log_f.write(f"[Val] Batch {batch_idx+1}/{len(val_loader)} | Folder: {folder_name} | batchLoss: {loss.item():.6f}\n")
                val_loss /= len(val_loader)
                print(f"Epoch {epoch+1}/{args.epochs}, Val Loss: {val_loss:.6f}")
                log_f.write(f"Epoch {epoch+1}/{args.epochs}, Val Loss: {val_loss:.6f}\n")
                with open(log_path, 'a') as f:
                  f.write(f"{epoch+1},{train_loss:.6f},{val_loss if args.val_data_dir else 0.0:.6f}\n")


                scheduler.step(val_loss)
                torch.cuda.empty_cache()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),  
                        'epoch': epoch
                    }, os.path.join(args.output_dir, f'bestmodel.pt'))

                    print(f"保存最佳模型，验证损失: {best_val_loss:.6f}")
            if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:        
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),  
                    'epoch': epoch
                       }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt'))



        print("done")

if __name__ == "__main__":
    args = parse_args()
    train(args)