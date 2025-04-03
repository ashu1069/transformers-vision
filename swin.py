import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_ (from original implementation)
# Let's implement DropPath simply here if timm library is not available.

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff embedding_dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


def window_partition(x, window_size):
    """
    Partition feature map into non-overlapping windows.
    Arguments:
        x (torch.Tensor): Input tensor of shape (B, H, W, C)
        window_size (int): Window size.
    Returns:
        torch.Tensor: Windows tensor of shape (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # contiguous() is important for performance and correctness after permute/transpose
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W, B):
    """
    Merge windows back to feature map.
    Arguments:
        windows (torch.Tensor): Windows tensor of shape (num_windows*B, window_size, window_size, C)
        window_size (int): Window size.
        H (int): Height of image.
        W (int): Width of image.
        B (int): Batch size.
    Returns:
        torch.Tensor: Merged tensor of shape (B, H, W, C)
    """
    # B_ = windows.shape[0] # num_windows * B
    # num_windows = H * W / window_size / window_size
    # B = B_ / num_windows
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ 
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both shifted and non-shifted window.

    Arguments:
        embedding_dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_attn_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, embedding_dim, window_size, num_attn_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.embedding_dim = embedding_dim
        # Please ensure window_size is a tuple
        self.window_size = (window_size, window_size) if isinstance(window_size, int) else window_size
        self.num_attn_heads = num_attn_heads
        head_dim = embedding_dim // num_attn_heads
        self.scale = head_dim ** -0.5

        # Define parameter table for relative position bias
        # Each dimension ranges from -window_size+1 to window_size-1 => 2*window_size-1 possibilities
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_attn_heads))

        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])

        # Create a grid of coordinates, shape: [2, Wh, Ww]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))

        # Flatten to [2, Wh*Ww]
        coords_flatten = torch.flatten(coords, 1)

        # Calculate relative coordinates: [2, Wh*Ww, Wh*Ww]
        # broadcasting: [2, Wh*Ww, 1] - [2, 1, Wh*Ww]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]

        # Permute to [Wh*Ww, Wh*Ww, 2]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()

        # Shift coordinate range from [-Wh+1, Wh-1] to [0, 2*Wh-2]
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1

        # Factorize the 2D index into a 1D index
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        # Sum along the last dimension to get the final 1D index: [Wh*Ww, Wh*Ww]
        relative_position_index = relative_coords.sum(-1)

        # Register buffer makes it part of the model state but not a trainable parameter
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(embedding_dim, embedding_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Initialize relative_position_bias_table properly
        torch.nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(embedding_dim=-1)

    def forward(self, x, mask=None):
        """
        Arguments:
            x (torch.Tensor): Input features with shape (num_windows*B, N, C)
                                N = window_size * window_size
            mask (torch.Tensor | None): Attention mask with shape (num_windows, N, N) or None.
                                      For SW-MSA, this prevents attention between disconnected regions.

        Returns:
            torch.Tensor: Output features with shape (num_windows*B, N, C)
        """
        B_, N, C = x.shape # B_ = num_windows * B

        # Calculate Q, K, V for all heads together: (B_, N, 3*C)
        qkv = self.qkv(x)

        # Reshape and permute for multi-head attention:
        # (B_, N, 3, num_attn_heads, C//num_attn_heads) -> (3, B_, num_attn_heads, N, C//num_attn_heads)
        qkv = qkv.reshape(B_, N, 3, self.num_attn_heads, C // self.num_attn_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # Separate Q, K, V: each shape (B_, num_attn_heads, N, C//num_attn_heads)
        q, k, v = qkv.unbind(0) # Alternative to qkv[0], qkv[1], qkv[2]

        # Scaled Dot-Product Attention: Q @ K.T / sqrt(d_k)
        q = q * self.scale

        # (B_, num_attn_heads, N, C//num_attn_heads) @ (B_, num_attn_heads, C//num_attn_heads, N) -> (B_, num_attn_heads, N, N)
        attn = (q @ k.transpose(-2, -1))

        # Add Relative Position Bias
        # We can fetch biases using the precomputed indices: (N, N) -> (N*N,) -> lookup -> (N*N, num_attn_heads)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]

        # Reshape to match attention matrix: (N, N, num_attn_heads) -> (num_attn_heads, N, N) for broadcasting
        relative_position_bias = relative_position_bias.view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        ).permute(2, 0, 1).contiguous()

        # Add bias to attention scores (broadcasting across batch dimension B_)
        attn = attn + relative_position_bias.unsqueeze(0) # Shape: (B_, num_attn_heads, N, N)

        # Apply Attention Mask (for SW-MSA)
        if mask is not None:
            # mask shape: (num_windows, N, N)
            # attn shape: (B_, num_attn_heads, N, N) = (B*num_windows, num_attn_heads, N, N)
            num_windows = mask.shape[0]

            # Add mask values. Need to reshape/expand mask to match attn:
            # (nW, 1, N, N) -> expands to (nW, num_attn_heads, N, N)
            # then repeat B times along the first embedding_dim? No, attn is already B*nW
            # Reshape attn to (B, nW, num_attn_heads, N, N) then add mask (1, nW, 1, N, N)
            # View attn as (B, num_windows, num_attn_heads, N, N)
            attn = attn.view(B_ // num_windows, num_windows, self.num_attn_heads, N, N)

            # Add mask (expand mask to match: 1, num_windows, 1, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0) # Shape: (B, num_windows, num_attn_heads, N, N)

            # Reshape back to (B_, num_attn_heads, N, N)
            attn = attn.view(-1, self.num_attn_heads, N, N)

            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        # Apply attention dropout
        attn = self.attn_drop(attn)

        # Multiply by V: (B_, num_attn_heads, N, N) @ (B_, num_attn_heads, N, C//num_attn_heads) -> (B_, num_attn_heads, N, C//num_attn_heads)
        x = (attn @ v)
        x = x.transpose(1, 2).reshape(B_, N, C)

        # Apply output projection and dropout
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block. Uses WindowAttention.

    Arguments:
        embedding_dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution (H, W).
        num_attn_heads (int): Number of attention heads.
        window_size (int): Window size. Default: 7
        shift_size (int): Shift size for SW-MSA. Default: 0 (no shift, W-MSA)
        mlp_ratio (float): Ratio of ff hidden embedding_dim to embedding_dim. Default: 4.0
        qkv_bias (bool, optional): If True, add bias to QKV projection. Default: True
        drop (float, optional): Dropout rate for Feed Forward. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """
    def __init__(self, embedding_dim, 
                 input_resolution, 
                 num_attn_heads, 
                 window_size=7, 
                 shift_size=0,
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 drop=0., 
                 attn_drop=0., 
                 drop_path=0.,
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_resolution = input_resolution
        self.num_attn_heads = num_attn_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # If window size is larger than input resolution, don't partition
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must be in [0, window_size)"

        self.norm1 = norm_layer(embedding_dim)
        self.attn = WindowAttention(
            embedding_dim, window_size=self.window_size, num_attn_heads=num_attn_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(embedding_dim)
        mlp_hidden_dim = int(embedding_dim * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, embedding_dim),
            nn.Dropout(drop)
        )

        # Pre-calculate attention mask for SW-MSA
        if self.shift_size > 0:
            H, W = self.input_resolution
            # Create a base mask canvas (1, H, W, 1)
            img_mask = torch.zeros((1, H, W, 1))

            # Define slices for shifting
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            
            # Assign unique IDs to different regions based on shift pattern
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # Partition the mask canvas into windows
            mask_windows = window_partition(img_mask, self.window_size) # Shape: (num_windows, Ws, Ws, 1)

            # Flatten the window dimension: (num_windows, Ws*Ws)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)

            # Calculate pair-wise differences in region IDs within each window
            # (nW, 1, Ws*Ws) - (nW, Ws*Ws, 1) -> (nW, Ws*Ws, Ws*Ws)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)

            # Create the actual mask: 0 for same region, -100 for different regions
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            # No mask needed for W-MSA (non-shifted)
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        """
        Forward pass for the Swin Transformer Block.
        Arguments:
            x (torch.Tensor): Input tensor of shape (B, L, C) where L = H * W.
        Returns:
            torch.Tensor: Output tensor of shape (B, L, C).
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"Input feature has wrong size L={L} vs H*W={H*W}"

        shortcut = x
        x = self.norm1(x)
        # Reshape to (B, H, W, C) for windowing operations
        x = x.view(B, H, W, C)

        # Cyclic Shift
        if self.shift_size > 0:
            # Shift the feature map to create a cyclic shift
            # Shifted tensor shape: (B, H, W, C)
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Window Partition
        # Partition the (shifted) feature map into windows: (nW*B, Ws, Ws, C)
        x_windows = window_partition(shifted_x, self.window_size)

        # Flatten window dimensions: (nW*B, Ws*Ws, C) ready for attention
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA / SW-MSA
        # Apply attention within windows, using the pre-calculated mask if SW-MSA
        # Input: (nW*B, Ws*Ws, C), Mask: (nW, Ws*Ws, Ws*Ws) or None
        # Output: (nW*B, Ws*Ws, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge Windows
        # Reshape back to (nW*B, Ws, Ws, C)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # Reverse the window partitioning: (B, H, W, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W, B)

        # Reverse Cyclic Shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        #  First Residual Connection & Feed Forward 
        x = shortcut + self.drop_path(x) # Apply drop path to the attention output

        # Apply Feed Forward (with LayerNorm, activation, dropout) and second residual connection
        x = x + self.drop_path(self.ff(self.norm2(x))) # Apply drop path to the Feed Forward output

        return x
