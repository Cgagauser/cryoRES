import torch
from einops import rearrange, repeat
from torch import nn
import math


def expand_to_batch(tensor, desired_size):
    tile = desired_size // tensor.shape[0]
    return repeat(tensor, 'b ... -> (b tile) ...', tile=tile)


def compute_mhsa(q, k, v, scale_factor=1, mask=None):
    # resulted shape will be: [batch, heads, tokens, tokens]
    scaled_dot_prod = torch.einsum('... i d , ... j d -> ... i j', q, k) / math.sqrt(q.shape[-1])

    if mask is not None:
        mask = (torch.bmm(mask.unsqueeze(-1),mask.unsqueeze(1))==0.).unsqueeze(1)
        scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -1e16)

    attention = torch.softmax(scaled_dot_prod, dim=-1)
    # calc result per head
    return torch.einsum('... i j , ... j d -> ... i d', attention, v),attention


class TransformerBlock(nn.Module):
    """
    Vanilla transformer block from the original paper "Attention is all you need"
    Detailed analysis: https://theaisummer.com/transformer/
    """

    def __init__(self, dim, heads=8, dim_head=None,
                 dim_linear_block=1024, dropout=0.1, activation=nn.GELU,
                 mhsa=None, prenorm=False):
        """
        Args:
            dim: token's vector length
            heads: number of heads
            dim_head: if none dim/heads is used
            dim_linear_block: the inner projection dim
            dropout: probability of droppping values
            mhsa: if provided you can change the vanilla self-attention block
            prenorm: if the layer norm will be applied before the mhsa or after
        """
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(dim=dim, heads=heads, dim_head=dim_head)
        self.mhca = MultiHeadCrossAttention(dim=dim, heads=heads, dim_head=dim_head)
        self.mhsa_s = MultiHeadSelfAttention(dim=dim, heads=heads, dim_head=dim_head)
        self.mhca_s = MultiHeadCrossAttention(dim=dim, heads=heads, dim_head=dim_head)
        self.prenorm = prenorm
        self.drop = nn.Dropout(dropout)
        self.drop_cross = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.norm_3 = nn.LayerNorm(dim)

        self.norm_1_s = nn.LayerNorm(dim)
        self.norm_2_s = nn.LayerNorm(dim)
        self.norm_3_s = nn.LayerNorm(dim)
    


        self.linear = nn.Sequential(
            nn.Linear(dim, dim_linear_block),
            activation(),  # nn.ReLU or nn.GELU
            nn.Dropout(dropout),
            nn.Linear(dim_linear_block, dim),
            nn.Dropout(dropout)
        )

        self.linear_s = nn.Sequential(
            nn.Linear(dim, dim_linear_block),
            activation(),  # nn.ReLU or nn.GELU
            nn.Dropout(dropout),
            nn.Linear(dim_linear_block, dim),
            nn.Dropout(dropout)
        )
   #    self.merge_map = nn.Linear(dim*2, dim)

    def forward(self, x, seq, mask):
        bs, map_len, dim = x.shape
        if self.prenorm:
            x = self.drop(self.mhsa(self.norm_1(x))) + x
            seq = self.drop(self.mhsa_s(self.norm_1_s(seq), mask)) + seq
            #x = self.drop_cross(self.mhca(self.norm_2(seq), x, x)) + x
            #seq = self.drop_cross(self.mhca_s(self.norm_2_s(x), seq, seq)) + seq
            mhca_out, _ = self.mhca(self.norm_2(seq), x, x)
            x = self.drop_cross(mhca_out) + x
            mhca_s_out, _ = self.mhca_s(self.norm_2_s(x), seq, seq)
            seq = self.drop_cross(mhca_s_out) + seq
            #zzy
            x = self.linear(self.norm_3(x)) + x
            seq = self.linear_s(self.norm_3_s(seq)) + seq
        else:
            x = self.norm_1(self.drop(self.mhsa(x)) + x)
            seq = self.norm_1_s(self.drop(self.mhsa_s(seq, mask)) + seq)
            #x = self.norm_2_s(self.drop_cross(self.mhca(x, seq, seq)) + x)
            #seq = self.norm_2(self.drop_cross(self.mhca_s(seq, x, x)) + seq)
            mhca_out, _ = self.mhca(x, seq, seq)
            x = self.norm_2_s(self.drop_cross(mhca_out) + x)
            mhca_s_out, _ = self.mhca_s(seq, x, x)
            seq = self.norm_2(self.drop_cross(mhca_s_out) + seq)
            ###zzy
            x = self.norm_3(self.linear(x) + x)
            seq = self.norm_3_s(self.linear_s(seq) + seq)
    #    x = self.merge_map(x.view(bs, -1, dim*2))
        return x, seq
    

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None):
        """
        Implementation of multi-head attention layer of the original transformer model.
        einsum and einops.rearrange is used whenever possible
        Args:
            dim: token's dimension, i.e. word embedding vector size
            heads: the number of distinct representations to learn
            dim_head: the dim of the head. In general dim_head<dim.
            However, it may not necessary be (dim/heads)
        """
        super().__init__()
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        _dim = self.dim_head * heads
        self.heads = heads
        self.to_qvk = nn.Linear(dim, _dim * 3, bias=False)
        self.W_0 = nn.Linear(_dim, dim, bias=False)
        self.scale_factor = self.dim_head ** -0.5

    def forward(self, x, mask=None):
        assert x.dim() == 3
        qkv = self.to_qvk(x)  # [batch, tokens, dim*3*heads ]

        # decomposition to q,v,k and cast to tuple
        # the resulted shape before casting to tuple will be: [3, batch, heads, tokens, dim_head]
        q, k, v = tuple(rearrange(qkv, 'b t (d k h ) -> k b h t d ', k=3, h=self.heads))

        out,_ = compute_mhsa(q, k, v, mask=mask, scale_factor=self.scale_factor)

        # re-compose: merge heads with dim_head
        out = rearrange(out, "b h t d -> b t (h d)")
        # Apply final linear transformation layer
        return self.W_0(out)
    

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None):
        """
        Implementation of multi-head attention layer of the original transformer model.
        einsum and einops.rearrange is used whenever possible
        Args:
            dim: token's dimension, i.e. word embedding vector size
            heads: the number of distinct representations to learn
            dim_head: the dim of the head. In general dim_head<dim.
            However, it may not necessary be (dim/heads)
        """
        super().__init__()
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        _dim = self.dim_head * heads
        self.heads = heads
        self.q = nn.Linear(dim, dim)
        self.to_vk = nn.Linear(dim * 2, _dim * 2, bias=False)
        self.W_0 = nn.Linear(_dim, dim, bias=False)
        self.scale_factor = self.dim_head ** -0.5

    def forward(self, q, k, v, mask=None):
        assert q.dim() == 3
        kv = self.to_vk(torch.cat([k,v], dim=-1))  # [batch, tokens, dim*3*heads ]

        # decomposition to q,v,k and cast to tuple
        # the resulted shape before casting to tuple will be: [3, batch, heads, tokens, dim_head]
        k, v = tuple(rearrange(kv, 'b t (d k h ) -> k b h t d ', k=2, h=self.heads))
        q = rearrange(self.q(q) , 'b t (d k h ) -> k b h t d ', k=1, h=self.heads)[0]

        out,attention= compute_mhsa(q, k, v, mask=mask, scale_factor=self.scale_factor)

        # re-compose: merge heads with dim_head
        out = rearrange(out, "b h t d -> b t (h d)")
        # Apply final linear transformation layer
        return self.W_0(out),attention
    

class Embeddings3D(nn.Module):
    def __init__(self, input_dim, embed_dim, cube_size, patch_size=16, dropout=0.1):
        super().__init__()
        self.n_patches = int((cube_size[0] * cube_size[1] * cube_size[2]) / (patch_size * patch_size * patch_size))
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embeddings = nn.Conv3d(in_channels=input_dim, out_channels=embed_dim,
                                          kernel_size=patch_size, stride=patch_size, bias=False)
        self.position_embeddings = AbsPositionalEncoding1D(self.n_patches, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.grid_shape = (
            cube_size[0] // patch_size,
            cube_size[1] // patch_size,
            cube_size[2] // patch_size
        )

        '''patch_indices = torch.tensor([
            (i, j, k)
            for i in range(self.grid_shape[0])  # x
            for j in range(self.grid_shape[1])  # y
            for k in range(self.grid_shape[2])  # z
        ]) '''
        #self.register_buffer("patch_indices", torch.tensor(patch_indices, dtype=torch.long)) 
        
    def forward(self, x):
        """
        x is a 5D tensor
        """
        x = rearrange(self.patch_embeddings(x), 'b d x y z -> b (x y z) d')
        embeddings = self.dropout(self.position_embeddings(x))
        #idx = 999
        #x, y, z = self.patch_indices[idx].tolist()
        #x_range = (x*self.patch_size, (x+1)*self.patch_size)
        #y_range = (y*self.patch_size, (y+1)*self.patch_size)
        #z_range = (z*self.patch_size, (z+1)*self.patch_size)
        #print(f"patch {idx} 在3D patch网格的索引坐标为: ({x}, {y}, {z})")
        #print(f"patch {idx} 在3D patch网格的坐标范围为: ({x_range}, {y_range}, {z_range})")




        return embeddings


class AbsPositionalEncoding1D(nn.Module):
    def __init__(self, tokens, dim):
        super(AbsPositionalEncoding1D, self).__init__()
        self.abs_pos_enc = nn.Parameter(torch.randn(1,tokens, dim))

    def forward(self, x):
        batch = x.size()[0]
        #print("batch:", batch)
        #print("x.shape:", x.shape)
        return x + expand_to_batch(self.abs_pos_enc, desired_size=batch)