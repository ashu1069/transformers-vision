import torch
import torch.nn as nn
import torch.nn.functional as F

from multihead_self_attention import MultiHeadSelfAttention

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embedding_dim):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.embedding_dim = embedding_dim

        # Convolutional layer to extract patches
        self.proj = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x) # Shape: (batch_size, embedding_dim, num_patches, num_patches)
        x = x.flatten(2) # Shape: (batch_size, embedding_dim, num_patches * num_patches)
        x = x.transpose(1, 2) # Shape: (batch_size, num_patches * num_patches, embedding_dim)
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, num_attn_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim) # We use layer norm to normalize the feature maps
        self.mha = MultiHeadSelfAttention(embedding_dim, num_attn_heads)

        # Dropout can be applied after MultiHead Self Attention and Feed Forward layers
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Attention part
        residual = x
        x = self.norm1(x)
        attn_output = self.mha(x)
        x = residual + attn_output # Skip connection

        # Feed Forward part
        residual = x
        x = self.norm2(x)   
        ff_output = self.ff(x)
        x = residual + ff_output # Skip connection
        return x
    
class VisionTransformer(nn.Module):
    def __init__(self,
                 image_size,
                 patch_size,
                 in_channels,
                 embedding_dim,
                 depth,
                 num_attn_heads,
                 ff_dim,
                 num_classes,
                 dropout=0.1):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.depth = depth
        self.num_attn_heads = num_attn_heads
        self.ff_dim = ff_dim
        self.num_classes = num_classes
        self.dropout = dropout
        
        # Patch embedding layer
        # This will convert the image into patches and project them into the embedding dimension
        self.patch_embedding = PatchEmbedding(self.image_size, self.patch_size, self.in_channels, self.embedding_dim)
        num_patches = self.patch_embedding.num_patches

        # CLS token (learnable parameter)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embedding_dim))

        # Positional embedding (learnable parameter): +1 for CLS token
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, self.embedding_dim))

        # Dropout for embeddings
        self.pos_drop = nn.Dropout(self.dropout)

        # Transformer Encoder layers
        self.blocks = nn.ModuleList([
            TransformerEncoder(self.embedding_dim, self.num_attn_heads, self.ff_dim, self.dropout) for _ in range(self.depth)
        ])

        # Final normalization layers
        self.norm = nn.LayerNorm(self.embedding_dim)
        self.head = nn.Linear(self.embedding_dim, self.num_classes)

    def forward(self, x):
        batch_size = x.shape[0]

        # Patch embedding
        x = self.patch_embedding(x) # Shape: (batch_size, num_patches, embedding_dim)

        # Preprend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # Shape: (batch_size, 1, embedding_dim)
        x = torch.cat((cls_tokens, x), dim=1) # Shape: (batch_size, num_patches + 1, embedding_dim)

        # Add positional embeddings
        x = x + self.pos_embedding
        x = self.pos_drop(x)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.norm(x)

        # Take the CLS token for classification
        cls_token_output = x[:, 0] # Shape: (batch_size, embedding_dim)
        output_logits = self.head(cls_token_output) # Shape: (batch_size, num_classes)

        return output_logits
