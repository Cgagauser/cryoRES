import numpy as np
# BLOSUM62 feature
AA = "ARNDCQEGHILKMFPSTWYV"
aa_to_idx = {aa: i for i, aa in enumerate(AA)}

BLOSUM62 = np.array([
    # A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V
    [ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0],  # A
    [-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3],  # R
    [-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3],  # N
    [-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3],  # D
    [ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],  # C
    [-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2],  # Q
    [-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2],  # E
    [ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3],  # G
    [-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3],  # H
    [-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3],  # I
    [-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1],  # L
    [-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2],  # K
    [-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1],  # M
    [-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1],  # F
    [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2],  # P
    [ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2],  # S
    [ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0],  # T
    [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3],  # W
    [-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1],  # Y
    [ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4]   # V
])

def blosum62_encoding(seq):
    return np.array([BLOSUM62[aa_to_idx.get(aa, 0)] for aa in seq])  # [L, 20]

def parse_aaindex1(file_path):
    props = {}
    with open(file_path, encoding='utf-8') as f:
        txt = f.read()
    blocks = txt.strip().split('//')
    for block in blocks:
        lines = block.strip().split('\n')
        if not lines or not lines[0].startswith('H'):
            continue
        code = lines[0][2:].strip() 
        desc = lines[1][2:].strip() if len(lines) > 1 and lines[1].startswith('D') else ''
        I_lines = [i for i, line in enumerate(lines) if line.startswith('I')]
        val_lines = []
        for idx in I_lines:
            val_lines.extend([lines[idx+1], lines[idx+2]])
        values = []
        for val_line in val_lines:
            values += [float(x) if x != 'NA' else 0.0 for x in val_line.strip().split()]
        if len(values) == 20:
            props[code] = (desc, np.array(values))
        else:
            print(f"警告：属性 {code} 不是20维，跳过")
    return props

aa_list = 'ARNDCQEGHILKMFPSTWYV'
aa_to_idx = {aa: i for i, aa in enumerate(aa_list)}

def get_seq_physicochem_feat(seq):
    """
    seq: str, 蛋白质序列
    props: 属性字典 parse_aaindex1() 输出
    selected_codes: 要用的属性编号列表
    返回: np.ndarray [L, N] (L=序列长度, N=属性数)
    """
    props = parse_aaindex1("/ziyingz/Programs/E3-CryoFold/AAindex.txt")
    selected_codes = [
        'KYTJ820101',   # 疏水性
        'GRAR740102',   # 极性
        'ZIMJ680104',   # 分子体积
        'JANJ780101',   # 最大表面可及性
        'BHAR880101',   # 柔性指数
        'CHOC760101',   # α螺旋倾向
        'CHOC760102',   # β-折叠倾向
        # 可选补充
        'FAUJ880109',  # 疏水自由能
        'FASG760101',  # 芳香族
        'DAYM780201',  # 氢键供体
        'KLEP840101',  # 侧链pKa
        'KIDA850101',   # 电子数（密度图相关）
    ]
    features = []
    for code in selected_codes:
        if code in props:
            _, arr = props[code]
        else:
            arr = np.zeros(20)  
        v = [arr[aa_to_idx[aa]] if aa in aa_to_idx else 0.0 for aa in seq]
        features.append(v)
    return np.array(features).T  

def normalize_to_minus_one_one(features):
    """将特征归一化到[-1, 1]范围"""
    min_val = np.min(features)
    max_val = np.max(features)
    
    # 避免除零错误
    if max_val == min_val:
        return np.zeros_like(features)
    
    # 归一化到[0, 1]
    normalized = (features - min_val) / (max_val - min_val)
    
    # 转换到[-1, 1]
    normalized = 6 * normalized - 3
    
    return normalized

def normalize_to_minus_one_one_v2(features):
    """将特征按维度归一化到[-1, 1]范围（简洁版本）
    
    Args:
        features: numpy数组，形状为 (n_samples, n_features) 或 (n_features,)
    
    Returns:
        归一化后的特征数组，每个维度独立归一化到[-1, 1]范围
    """
    # 沿着样本维度计算每个特征的最小值和最大值
    min_val = np.min(features, axis=0, keepdims=True)
    max_val = np.max(features, axis=0, keepdims=True)
    
    # 计算范围，避免除零
    range_val = max_val - min_val
    range_val = np.where(range_val == 0, 1, range_val)  # 将0替换为1避免除零
    
    # 归一化到[0, 1]然后转换到[-1, 1]
    normalized = (features - min_val) / range_val
    normalized = 2 * normalized - 1
    
    return normalized

    
def get_seq_blosum_physicochem_feat(seq):
    """
    seq: str，标准20AA单字母序列
    aaindex_path: AAindex1数据库文件路径
    selected_codes: AAindex特征编号列表
    返回: np.ndarray [L, 20+N]
    """
    # BLOSUM62编码
    blosum_feat = blosum62_encoding(seq)  # [L, 20]
    physicochem_feat = get_seq_physicochem_feat(seq)  # [L, N]
    blosum62_normalized = normalize_to_minus_one_one_v2(blosum_feat)
    physicochemical_normalized = normalize_to_minus_one_one_v2(physicochem_feat)
    # print(f"\n归一化后BLOSUM62特征范围: [{blosum62_normalized.min():.3f}, {blosum62_normalized.max():.3f}]")
    # print(f"归一化后理化特性特征范围: [{physicochemical_normalized.min():.3f}, {physicochemical_normalized.max():.3f}]")
    # 拼接
    out = np.concatenate([blosum62_normalized, physicochemical_normalized], axis=1)  # [L, 20+N]
    out = np.array(out)  # 如果它是其他类型，可以转换为 np.ndarray
    return out




if __name__ == "__main__":
    seq = "MKTLLILAVVLAF"
    feat = get_seq_blosum_physicochem_feat(seq)
    print(feat.shape)