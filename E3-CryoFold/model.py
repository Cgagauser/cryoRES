import esm
import torch
import torch.nn as nn
from src.transformer_module import Embeddings3D, TransformerBlock, MultiHeadCrossAttention

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout, extract_layers, dim_linear_block):
        super().__init__()
        self.layer = nn.ModuleList()
        self.extract_layers = extract_layers

        self.block_list = nn.ModuleList()
        for _ in range(num_layers):
            self.block_list.append(
                TransformerBlock(dim=embed_dim, heads=num_heads, dim_linear_block=dim_linear_block, dropout=dropout,
                                 prenorm=False))

    def forward(self, x, seq, mask=None):
        for layer_block in self.block_list:
            x, seq = layer_block(x, seq, mask)
        return x, seq

class ImprovedResidualBlock(nn.Module):
    """改进的残差块，添加梯度缩放"""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),  # 使用GELU替代ReLU
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)  # 可学习的残差缩放
        
    def forward(self, x):
        return self.norm(x + self.scale * self.block(x))

class StableAttentionPooling(nn.Module):
    """稳定的注意力池化"""
    def __init__(self, dim, temperature=1.0):
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.LayerNorm(dim // 4),  # 添加LayerNorm
            nn.GELU(),
            nn.Linear(dim // 4, 1)
        )
        
    def forward(self, x):
        # x: [B, P, D]
        attn_scores = self.attention(x) / self.temperature  # [B, P, 1]
        weights = torch.softmax(attn_scores, dim=1)
        
        # 添加权重正则化，防止注意力过于集中
        entropy_reg = -torch.mean(weights * torch.log(weights + 1e-8))
        
        pooled = torch.sum(x * weights, dim=1)  # [B, D]
        return pooled, weights, entropy_reg

class SimplifiedDecoder(nn.Module):
    """简化的多尺度解码器"""
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        
        # 简化为两个分支，减少参数
        self.branch1 = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, output_dim // 2)
        )
        
        self.branch2 = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim // 2)
        )
        
        # 简化融合层
        self.fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
        
    def forward(self, x):
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        combined = torch.cat([branch1_out, branch2_out], dim=-1)
        return self.fusion(combined)

class StablePatchClassifier(nn.Module):
    """稳定的Patch分类器，防止模型坍塌"""
    def __init__(self, input_dim=768, hidden_dim=1024, num_classes=1000, dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # 第一阶段：渐进式特征扩展
        self.feature_expansion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 第二阶段：减少残差块数量，添加skip connection
        self.deep_processing = nn.ModuleList([
            ImprovedResidualBlock(hidden_dim, dropout) for _ in range(2)  # 从4减少到2
        ])
        
        # 第三阶段：简化的多尺度处理
        self.multiscale_decoder = SimplifiedDecoder(
            input_dim=hidden_dim, 
            output_dim=hidden_dim,
            dropout=dropout
        )
        
        # 第四阶段：稳定的注意力机制
        self.attention_pooling = StableAttentionPooling(hidden_dim)
        
        # 第五阶段：简化的分类头
        self.patch_wise_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),  # 渐减dropout
            
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # 添加温度缩放参数
        self.temperature = nn.Parameter(torch.ones(1))
        
        # 改进的权重初始化
        self._initialize_weights()
        
    def _initialize_weights(self):
        """改进的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用Kaiming初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, return_attention=False):
        """
        Args:
            x: [B, P, D] - Batch, Patches, Dimensions
            return_attention: 是否返回注意力权重
        Returns:
            patch_logits: [B, P, num_classes] - 每个patch的分类概率
            attention_weights: [B, P, 1] - 注意力权重 (可选)
            entropy_reg: 注意力熵正则化项 (可选)
        """
        batch_size, num_patches, input_dim = x.shape
        
        # 第一阶段：特征扩展
        x_expanded = self.feature_expansion(x)  # [B, P, hidden_dim]
        
        # 第二阶段：深度处理 (添加skip connection)
        x_deep = x_expanded
        residual = x_expanded
        for i, residual_block in enumerate(self.deep_processing):
            x_deep = residual_block(x_deep)
            if i == 0:  # 第一个block后保存残差
                residual = x_deep
        
        # 添加长程skip connection
        x_deep = x_deep + 0.1 * residual
        
        # 第三阶段：多尺度解码
        x_multiscale = self.multiscale_decoder(x_deep)  # [B, P, hidden_dim]
        
        # 第四阶段：稳定的全局上下文
        global_context, attention_weights, entropy_reg = self.attention_pooling(x_multiscale)
        
        # 将全局上下文广播到每个patch
        global_context_expanded = global_context.unsqueeze(1).expand(-1, num_patches, -1)
        
        # 结合局部和全局特征
        combined_features = torch.cat([x_multiscale, global_context_expanded], dim=-1)
        
        # 第五阶段：最终分类 (添加温度缩放)
        patch_logits = self.patch_wise_classifier(combined_features)
        patch_logits = patch_logits / self.temperature  # 温度缩放
        
        if return_attention:
            return patch_logits, attention_weights, entropy_reg
        return patch_logits



class CryoFold(nn.Module):
    def __init__(self, img_shape=(360, 360, 360), input_dim=1, output_dim=4, embed_dim=768, patch_size=36,
                num_heads=6, dropout=0.1, ext_layers=[3, 6, 9, 12], norm="instance",
                dim_linear_block=3072, decoder_dim=256):
        super().__init__()
        self.num_layers = 8   #origin 8 
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.img_shape = img_shape
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.ext_layers = ext_layers
        self.decoder_dim = decoder_dim

        esm_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        self.esm = esm_model
        self.norm = nn.BatchNorm3d if norm == 'batch' else nn.InstanceNorm3d
        self.embed = Embeddings3D(input_dim=input_dim, embed_dim=embed_dim, cube_size=img_shape,
                                  patch_size=patch_size, dropout=dropout)
        self.position_emb = nn.Embedding(num_embeddings=30000, embedding_dim=embed_dim)
        #self.token_embed = nn.Embedding(num_embeddings=33, embedding_dim=embed_dim, padding_idx=alphabet.padding_idx)
        self.chain_embed = nn.Embedding(num_embeddings=1000, embedding_dim=embed_dim, padding_idx=0)
    
        self.transformer = TransformerEncoder(embed_dim, num_heads, self.num_layers, dropout, ext_layers,
                                              dim_linear_block=dim_linear_block)
                                              
        self.patch_classifier = StablePatchClassifier(
            input_dim=embed_dim, 
            hidden_dim=2048,  # 增加到2048
            num_classes=1000,
            dropout=dropout
        )

        #self.out = nn.Linear(embed_dim, 12)
        #self.to_hV = nn.Linear(embed_dim, decoder_dim)
        #self.decoder_struct = StructDecoder(8, decoder_dim, 1)
        #self.atom_norm = nn.LayerNorm(12)
        self.cross_attn = MultiHeadCrossAttention(embed_dim, num_heads)

    def normalize_features_per_dim(self, features, target_min=-1, target_max=1):
        """逐特征维度归一化"""
        # features: [B, L, D]
        B, L, D = features.shape
        # Reshape to [B*L, D] for per-dimension normalization
        features_flat = features.view(-1, D)
        # 计算每个维度的最小值和最大值
        min_vals = features_flat.min(dim=0, keepdim=True)[0]  # [1, D]
        max_vals = features_flat.max(dim=0, keepdim=True)[0]  # [1, D]
        # 归一化
        range_vals = max_vals - min_vals
        range_vals = torch.clamp(range_vals, min=1e-6)  # 避免除零
        normalized = (features_flat - min_vals) / range_vals
        normalized = normalized * (target_max - target_min) + target_min
        # Reshape back
        return normalized.view(B, L, D)
        
    def get_esm_feature_sliding(self, seq, window_size=1000, stride=800, layer=12):
            B, L = seq.shape
            device = seq.device
            embed = 480
            feature_sum = torch.zeros((B, L, embed), device=device)
            feature_count = torch.zeros((B, L), device=device)
            
            # 调试信息
            #print(f"处理序列长度: {L}, 窗口大小: {window_size}, 步长: {stride}")
            with torch.no_grad():
                for start in range(0, L, stride):
                    end = min(start + window_size, L)
                    
                    # 取窗口片段
                    subseq = seq[:, start:end]  # [B, window]
                    out = self.esm(subseq, repr_layers=[layer])['representations'][layer]  # [B, window, embed]
                    
                    # 累加特征（重叠部分会自动累加）
                    feature_sum[:, start:end, :] += out
                    feature_count[:, start:end] += 1
                    
                    if end == L:
                        break
                
                # 检查窗口覆盖情况
                min_count = feature_count.min().item()
                max_count = feature_count.max().item()
                # print(f"窗口覆盖次数: 最小={min_count}, 最大={max_count}")
                
                # 计算平均特征（重叠部分通过除以count自动平均）
                feature_avg = feature_sum / feature_count.unsqueeze(-1)  # [B, L, embed]
                
                # 打印归一化前的统计信息
                # print(f"归一化前ESM特征: 范围=[{feature_avg.min():.3f}, {feature_avg.max():.3f}], "
                      # f"均值={feature_avg.mean():.3f}, 标准差={feature_avg.std():.3f}")
                
                # 进行每维归一化到[-3, 3]
                feature_normalized = self.normalize_features_per_dim(feature_avg, target_min=-1, target_max=1)
                
                # 打印归一化后的统计信息
                # print(f"归一化后ESM特征: 范围=[{feature_normalized.min():.3f}, {feature_normalized.max():.3f}], "
                      # f"均值={feature_normalized.mean():.3f}, 标准差={feature_normalized.std():.3f}")
                
                return feature_normalized
            
            
    # def get_esm_feature_sliding(self, seq, window_size=1000, stride=800, layer=12):
        # B, L = seq.shape
        # device = seq.device
        # embed = 480
        # feature_sum = torch.zeros((B, L, embed), device=device)
        # feature_count = torch.zeros((B, L), device=device)
        # for start in range(0, L, stride):
            # end = min(start + window_size, L)
            # # 取窗口片段
            # subseq = seq[:, start:end]  # [B, window]
            # out = self.esm(subseq, repr_layers=[layer])['representations'][layer]  # [B, window, embed]
            # feature_sum[:, start:end, :] += out
            # feature_count[:, start:end] += 1
            # if end == L:
                # break
        # feature_avg = feature_sum / feature_count.unsqueeze(-1)  # [B, L, embed]
        # return feature_avg

    def forward(self, x, seq, seq_pos, chain_encoding,physicochem_feat, mask=None):
        batch_size = x.shape[0]
        if x.dim() == 4:
            x = x.unsqueeze(1) 
        _, length = seq.shape
        # print("="*50)
        #print(f"输入 x 形状: {x.shape}") #1,1,360,360,360
        # print(f"输入 seq 形状: {seq.shape}") #1,974
        transformer_input = self.embed(x)
        seq = self.get_esm_feature_sliding(seq, window_size=1000, stride=500, layer=12)
        # print(f"ESM特征最小值: {seq.min():.3f}")
        # print(f"ESM特征最大值: {seq.max():.3f}")
        seq= torch.cat([seq, physicochem_feat], axis=2)
        seq=seq.float() 
        position_embeddings = self.position_emb(seq_pos)  # [B, L, embed_dim]
        chain_embeddings = self.chain_embed(chain_encoding)  # [B, L, embed_dim]
        seq = seq + position_embeddings + chain_embeddings
        
        #seq = seq + self.chain_embed(chain_encoding) + self.position_emb(seq_pos) 
        protein, seq = self.transformer(transformer_input, seq, mask.float())
        y,attention = self.cross_attn(seq, protein, protein)
        patch_logits = self.patch_classifier(y)  # [B, L, 1000]
        log_probs = torch.log_softmax(patch_logits, dim=-1)  # [B, L, 1000]

        first_pos_pred = torch.argmax(log_probs[0, 0, :])
        mid_pos_pred = torch.argmax(log_probs[0, length//2, :]) if length > 1 else first_pos_pred
        last_pos_pred = torch.argmax(log_probs[0, -1, :])
        # print(f"第1个位置预测: {first_pos_pred}, 中间位置预测: {mid_pos_pred}, 最后位置预测: {last_pos_pred}")
        # return log_probs
        pred_info = {
        'first_pos': first_pos_pred.item(),
        'mid_pos': mid_pos_pred.item(), 
        'last_pos': last_pos_pred.item()
        }
    
        return log_probs, pred_info


        '''patch_score = self.patchclassifier(y)  #[B,L] 
        prob = torch.sigmoid(patch_score)  # [B, 1000]
        print(patch_score.shape)
        print(prob)
        h_V = self.to_hV(y)  #维度压缩 480 to 128
        #print(f"h_V 形状: {h_V.shape}") #1,974,128
        X = self.atom_norm(self.out(y)).view(batch_size, length, 4, 3)[mask]
        #print(f"最终输出 X 形状: {X.shape}") #974,4,3
        batch_id = torch.arange(x.shape[0]).view(-1, 1).repeat(1, length).to(x.device)[mask]
        # print(f"batch_id 形状: {batch_id.shape}")
        # print(f"batch_id 的第一个元素: {batch_id[0]}")
        #print(f"batch_id 的第500个元素: {batch_id[500]}") 
        chain_encoding = chain_encoding[mask]
        # print(f"chain_encoding 形状: {chain_encoding.shape}") #974
        # print(f"chain_encoding 的第一个元素: {chain_encoding[0]}") #1
        # print("="*50)
        X_pred, all_preds = self.decoder_struct.infer_X(X, h_V[mask], batch_id, chain_encoding, 30, virtual_frame_num=3)
        # print(f"最终输出 X_pred 形状: {X_pred.shape}") #974,4,3
        # print(f"最终输出 all_preds 形状: {len(all_preds)}") #8
        # print(f"all_preds 的第一个元素形状: {all_preds[0].shape if isinstance(all_preds[0], torch.Tensor) else type(all_preds[0])}") #974,4,3
        return X_pred, all_preds'''
    
    def infer(self, cryo_map, seq, chain_encoding, physicochem_feat,max_len=1000):
        self.eval()
        print(f"cryo_map shape: {cryo_map.shape}") #360,360,360
        #print(f"seq shape: {seq.shape}") #974
        seq=seq.unsqueeze(0)
        physicochem_feat=physicochem_feat.unsqueeze(0)
        print(f"chain_encoding shape: {chain_encoding.shape}")     #974
        seq_pos = torch.arange(seq.shape[0], device=cryo_map.device)
        #seq, chain_encoding, seq_pos= map(lambda x: x[:max_len].unsqueeze(0), [seq, chain_encoding.long(), seq_pos])
        #print(seq)
        protein_data = cryo_map.reshape(1, 1, *cryo_map.shape)
        mask = torch.ones_like(seq).bool()
        print(f"seq2 shape: {seq.shape}") #1,974
        #X_pred, all_preds = self.forward(protein_data, seq, seq_pos, chain_encoding, mask)
        #return X_pred, all_preds
        X_pred = self.forward(protein_data, seq, seq_pos, chain_encoding,physicochem_feat, mask)
        return X_pred
