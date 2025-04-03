import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_attn_heads):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_attn_heads = num_attn_heads
        self.head_dim = embedding_dim // num_attn_heads

        # Linear layers for Q,K,V projections for all heads together
        self.qkv_proj = nn.Linear(embedding_dim, 3 * embedding_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.shape

        # Linearly project Q, K, V and split into multiple heads
        qkv = self.qkv_proj(x) # Shape: (batch_size, seq_length, 3 * embedding_dim)
        qkv = qkv.reshape(batch_size, seq_length, 3, self.num_attn_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # Shape: (3, batch_size, num_attn_heads, seq_length, head_dim)

        # Split into Q, K, V
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled Dot Product Attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        scale_factor = math.sqrt(self.head_dim)
        attn_scores = attn_scores / scale_factor

        if mask is not None:
            # We should ensure that the mask has compatible shape, e.g., (batch_size, 1, 1, seq_length)
            # and add a large negative value where mask is True
            # This will ensure that the softmax will ignore these positions
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Now, apply softmax to get attention weights/probabilities
        attn_probs = F.softmax(attn_scores, dim=-1) # Shape: (batch_size, num_attn_heads, seq_length, seq_length)

        # Apply attention probabilities to V
        # This will give us the final attention output
        attn_output = torch.matmul(attn_probs, v) # Shape: (batch_size, num_attn_heads, seq_length, head_dim)

        # Now, concatenate the heads and project back to the original embedding dimension
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous() # Shape: (batch_size, seq_length, num_attn_heads, head_dim)
        attn_output = attn_output.view(batch_size, seq_length, self.embedding_dim) # Shape: (batch_size, seq_length, embedding_dim)

        # Apply the final linear projection
        output = self.out_proj(attn_output) # Shape: (batch_size, seq_length, embedding_dim)

        return output