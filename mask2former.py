# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionEmbeddingSine(nn.Module):
    """
    2D Sine-Cosine positional encoding.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        """
        Arguments:
            x (Tensor): Input tensor (e.g., image features) of shape [B, C, H, W]
            mask (Tensor | None): Boolean mask of shape [B, H, W] where True indicates padding.
        Returns:
            pos (Tensor): Positional encoding of shape [B, num_pos_feats * 2, H, W]
        """
        if mask is None:
            mask = torch.zeros(x.shape[0], x.shape[2], x.shape[3], device=x.device, dtype=torch.bool)

        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32) # Cumulative sum along Height
        x_embed = not_mask.cumsum(2, dtype=torch.float32) # Cumulative sum along Width

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)

        # i // 2 calculation for dimensions
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t # Shape: [B, H, W, num_pos_feats]
        pos_y = y_embed[:, :, :, None] / dim_t # Shape: [B, H, W, num_pos_feats]

        # Apply sin/cos to alternating dimensions
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        # Concatenate x and y positional encodings
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2) # Shape: [B, num_pos_feats*2, H, W]
        return pos

class SimpleBackbone(nn.Module):
    """ Outputs dummy features at different scales, we can use Swin Transformer block here as well. """
    def __init__(self, out_channels=[64, 128, 256]):
        super().__init__()
        self.out_channels = out_channels

        # Example layers (replace with actual ResNet/Swin stages)
        self.conv1 = nn.Conv2d(3, out_channels[0], kernel_size=3, stride=2, padding=1) # Stride 2 -> 1/2 res
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels[0], out_channels[1], kernel_size=3, stride=2, padding=1) # Stride 4 -> 1/4 res
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channels[1], out_channels[2], kernel_size=3, stride=2, padding=1) # Stride 8 -> 1/8 res
        self.relu3 = nn.ReLU()
        print(f"Initialized SimpleBackbone with out_channels={out_channels}")

    def forward(self, x):
        features = {}
        f1 = self.relu1(self.conv1(x))
        features['res2'] = f1 # Example name for 1/2 scale
        f2 = self.relu2(self.conv2(f1))
        features['res3'] = f2 # Example name for 1/4 scale
        f3 = self.relu3(self.conv3(f2))
        features['res4'] = f3 # Example name for 1/8 scale
        return features

class SimplePixelDecoder(nn.Module):
    """ Combines multi-scale features into mask features and pixel decoder features. """
    def __init__(self, backbone_channels, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        # Simple lateral + output convolutions (like basic FPN)
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        # Process features from higher res to lower res
        # Use reversed channels list assuming features['res2'], features['res3'], features['res4']
        reversed_channels = backbone_channels[::-1] # [256, 128, 64]

        # Lateral connections to reduce channels to embed_dim
        for channels in reversed_channels:
            self.lateral_convs.append(nn.Conv2d(channels, embed_dim, kernel_size=1))

        # Output convolutions to refine features after merging
        # We'll have len(reversed_channels) feature maps to combine
        for _ in range(len(reversed_channels)):
             self.output_convs.append(nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
                nn.ReLU(),
            ))

        # Final projection layers
        # For mask features (used to generate final masks)
        self.mask_features_head = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        # For pixel decoder features (fed into transformer cross-attention)
        self.pixel_decoder_head = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        print(f"Initialized SimplePixelDecoder targeting embed_dim={embed_dim}")


    def forward(self, backbone_features):
        # Process features from lowest res (last backbone stage) to highest
        # Assume backbone_features is like {'res4': f_low, 'res3': f_mid, 'res2': f_high}
        feature_keys = sorted(backbone_features.keys(), reverse=True) # ['res4', 'res3', 'res2']
        lat_features = []
        # Apply lateral connections
        for i, key in enumerate(feature_keys):
            lat_features.append(self.lateral_convs[i](backbone_features[key]))

        # Top-down pathway with summation
        merged_features = []
        prev_feature = lat_features[0] # Lowest resolution lateral feature
        merged_features.append(self.output_convs[0](prev_feature))

        for i in range(1, len(lat_features)):
            # Upsample previous feature and add to current lateral feature
            upsampled_prev = F.interpolate(prev_feature, size=lat_features[i].shape[-2:], mode='nearest')
            merged = lat_features[i] + upsampled_prev
            prev_feature = self.output_convs[i](merged) # Apply output conv
            merged_features.append(prev_feature)

        # Use the highest resolution merged feature map (last element after reversal)
        # Or potentially combine them - let's use the highest res one for simplicity
        final_feature_map = merged_features[-1] # Highest resolution feature map

        mask_features = self.mask_features_head(final_feature_map)
        pixel_decoder_output = self.pixel_decoder_head(final_feature_map)

        # pixel_decoder_output is used for cross-attention (memory)
        # mask_features is used with query embeddings to generate final segmentation masks
        return pixel_decoder_output, mask_features


class MaskedCrossAttention(nn.Module):
    """ Simplified Cross-Attention that incorporates an external mask. """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (embed_dim // num_heads) ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_pos_embed, query_pos_embed, attention_mask=None):
        """
        Arguments:
            query (Tensor): Query embeddings [B, num_queries, C]
            key (Tensor): Key embeddings (flattened pixel features) [B, H*W, C]
            value (Tensor): Value embeddings (flattened pixel features) [B, H*W, C]
            key_pos_embed (Tensor): Positional encoding for keys [B, H*W, C]
            query_pos_embed (Tensor): Positional encoding for queries [B, num_queries, C]
            attention_mask (Tensor | None): Mask derived from predicted segmentation masks.
                                          Shape [B, num_queries, H*W]. Values should be
                                          0 (attend) or -inf (mask out).
        Returns:
            Tensor: Output features [B, num_queries, C]
        """
        B, nq, C = query.shape
        B, L, C = key.shape # L = H * W

        # Apply positional embeddings before projection (common practice)
        query = query + query_pos_embed
        key = key + key_pos_embed

        # Project Q, K, V
        q = self.q_proj(query).view(B, nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B, nH, nq, C/nH
        k = self.k_proj(key).view(B, L, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B, nH, L, C/nH
        v = self.v_proj(value).view(B, L, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B, nH, L, C/nH

        # Calculate attention scores
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale # B, nH, nq, L

        # --- Apply the external attention mask (Masked Attention Logic) ---
        if attention_mask is not None:
            # Expand mask from [B, nq, L] to match attn_scores [B, nH, nq, L]
            attn_scores = attn_scores + attention_mask.unsqueeze(1) # Add -inf where mask indicates

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        output = (attn_probs @ v).permute(0, 2, 1, 3).reshape(B, nq, C) # B, nq, C

        # Output projection
        output = self.out_proj(output)
        return output

class Mask2FormerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # Query Self-Attention
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # Masked Cross-Attention
        self.cross_attn = MaskedCrossAttention(embed_dim, num_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feed-Forward Network
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.activation = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, query, query_pos_embed, key, key_pos_embed, value, attention_mask=None):
        """
        Arguments:
            query: [B, num_queries, C]
            query_pos_embed: [B, num_queries, C]
            key: (Pixel features) [B, H*W, C]
            key_pos_embed: [B, H*W, C]
            value: (Pixel features, often same as key) [B, H*W, C]
            attention_mask: Mask for cross-attention [B, num_queries, H*W]
        """
        # 1. Self-Attention on queries
        q = k_sa = query + query_pos_embed # Add pos embedding for self-attn Q, K
        self_attn_output, _ = self.self_attn(q, k_sa, value=query) # Use original query for V
        query = query + self.dropout1(self_attn_output)
        query = self.norm1(query)

        # 2. Masked Cross-Attention
        cross_attn_output = self.cross_attn(
            query=query, # Use normalized query from self-attn
            key=key,     # Pixel features
            value=value, # Pixel features
            key_pos_embed=key_pos_embed,
            query_pos_embed=query_pos_embed, # Add pos embedding again for cross-attn Q
            attention_mask=attention_mask
        )
        query = query + self.dropout2(cross_attn_output)
        query = self.norm2(query)

        # 3. Feed-Forward Network
        ffn_output = self.linear2(self.dropout3(self.activation(self.linear1(query))))
        query = query + self.dropout4(ffn_output)
        query = self.norm3(query)

        return query

class Mask2FormerTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, embed_dim):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        # Layer for predicting masks from intermediate layers (needed for iterative masking)
        # Simplified: we'll predict masks *after* all layers for this example
        # self.mask_predictor = SomeMaskPredictor(...)

    def forward(self, query, query_pos_embed, key, key_pos_embed, value, intermediate_mask_preds=None):
        """
        Arguments:
            query: Initial query embeddings [B, num_queries, C]
            query_pos_embed: Positional embedding for queries [B, num_queries, C]
            key: Pixel features [B, H*W, C]
            key_pos_embed: Positional encoding for pixel features [B, H*W, C]
            value: Pixel features (usually same as key) [B, H*W, C]
            intermediate_mask_preds: (Optional) Predicted masks [B, num_queries, H, W]
                                      used to create attention_mask. If None, no masking.
        Returns:
            output_query (Tensor): Output query embeddings after all layers [B, num_queries, C]
        """
        output_query = query
        L = key.shape[1] # H*W

        for i, layer in enumerate(self.layers):
            # Create Attention Mask
            # In a real implementation, masks predicted *from the output of layer i-1*
            # would be used to generate the mask for *layer i*.
            # Simplified: Use the single provided `intermediate_mask_preds` for all layers,
            # or no mask if None.
            attention_mask = None
            if intermediate_mask_preds is not None:
                # Ensure masks have the right spatial dimensions (match key's H*W)
                B, nq, H_mask, W_mask = intermediate_mask_preds.shape
                # Interpolate masks if necessary (e.g., if key features have different H, W)
                # Assuming they match L for simplicity here.
                # Flatten masks: [B, nq, H*W]
                mask_flat = intermediate_mask_preds.view(B, nq, H_mask * W_mask)
                # Ensure mask shape matches L = key.shape[1]
                if mask_flat.shape[-1] != L:
                     # This indicates a mismatch, needs proper handling (interpolation/resizing)
                     # For simplicity, let's assume they match or skip masking
                     print(f"Warning: Mask shape {mask_flat.shape} mismatch with key shape L={L}. Skipping mask.")
                else:
                     # Convert mask probabilities to -inf/0 mask for attention
                     # Thresholding example: Attend only where mask prob > 0.5
                     attention_mask = (mask_flat < 0.5).float() * -1e9 # Use large negative number
                     # Ensure it's compatible: [B, nq, L]

            output_query = layer(
                query=output_query,
                query_pos_embed=query_pos_embed,
                key=key,
                key_pos_embed=key_pos_embed,
                value=value,
                attention_mask=attention_mask
            )
            # Intermediate mask prediction would happen here in iterative versions
            # intermediate_mask_preds = self.mask_predictor(output_query, mask_features_from_pixel_decoder)

        return output_query

class Mask2Former(nn.Module):
    def __init__(self, num_classes, num_queries=100, embed_dim=256, num_heads=8,
                 decoder_layers=6, dim_feedforward=2048, backbone_channels=[64, 128, 256],
                 dropout=0.1):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim

        # Backbone
        self.backbone = SimpleBackbone(out_channels=backbone_channels)

        # Pixel Decoder
        self.pixel_decoder = SimplePixelDecoder(backbone_channels, embed_dim)

        # Positional Encodings
        # For pixel decoder output (used in cross-attention)
        self.pixel_pos_embed = PositionEmbeddingSine(embed_dim // 2, normalize=True)

        # Learnable query embeddings (object queries)
        self.query_embed = nn.Embedding(num_queries, embed_dim)

        # Learnable positional embedding for queries (can be added or used as init)
        self.query_pos_embed = nn.Embedding(num_queries, embed_dim)

        # Transformer Decoder
        decoder_layer = Mask2FormerDecoderLayer(embed_dim, num_heads, dim_feedforward, dropout)
        self.decoder = Mask2FormerTransformerDecoder(decoder_layer, decoder_layers, embed_dim)

        # Prediction Heads
        # Class prediction head
        self.class_head = nn.Linear(embed_dim, num_classes + 1) # +1 for no-object class

        # Mask prediction head (simple MLP to project query embed)
        self.mask_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        print("Initialized Mask2Former Model")

    def forward(self, x):
        """
        Arguments:
            x (torch.Tensor): Input image batch [B, 3, H_img, W_img]
        Returns:
            dict: {'pred_logits': [B, num_queries, N_classes+1],
                   'pred_masks': [B, num_queries, H_mask, W_mask]}
        """
        B = x.shape[0]
        H_img, W_img = x.shape[-2:]

        # Backbone -> Multi-scale features
        backbone_features = self.backbone(x)

        # Pixel Decoder -> High-res features for masks & cross-attention
        # pixel_decoder_output: features for cross-attn memory [B, C, H_pix, W_pix]
        # mask_features: features projected by mask head [B, C, H_mask, W_mask] (often same res)
        pixel_decoder_output, mask_features = self.pixel_decoder(backbone_features)

        # Prepare inputs for Transformer Decoder
        # Flatten pixel features and generate positional embeddings
        B, C, H_pix, W_pix = pixel_decoder_output.shape
        pixel_features_flat = pixel_decoder_output.flatten(2).transpose(1, 2) # B, H*W, C
        pixel_pos_embed_2d = self.pixel_pos_embed(pixel_decoder_output) # B, C, H, W
        pixel_pos_embed_flat = pixel_pos_embed_2d.flatten(2).transpose(1, 2) # B, H*W, C

        # Prepare query embeddings
        query_embeds_init = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1) # B, nq, C
        query_pos_embeds = self.query_pos_embed.weight.unsqueeze(0).repeat(B, 1, 1) # B, nq, C

        # Generate initial mask prediction for first layer masking
        # In a real model, this might be zero, uniform, or from a simple predictor.
        # Here, we'll skip iterative masking and predict masks *only at the end*.
        # Set intermediate_mask_preds=None for the decoder forward pass.
        intermediate_mask_preds_for_attn = None

        # Transformer Decoder
        # Output: Query embeddings refined by attending to pixel features [B, num_queries, C]
        decoder_output_queries = self.decoder(
            query=query_embeds_init,
            query_pos_embed=query_pos_embeds,
            key=pixel_features_flat,
            key_pos_embed=pixel_pos_embed_flat,
            value=pixel_features_flat, # Key and Value are the same here
            intermediate_mask_preds=intermediate_mask_preds_for_attn # None in this simplified version
        )
        # Class prediction: [B, num_queries, num_classes+1]
        class_logits = self.class_head(decoder_output_queries)

        # Mask prediction:
        # Project decoder output queries: [B, num_queries, C]
        query_mask_embeds = self.mask_head(decoder_output_queries)

        # Generate masks by dot product with mask_features from pixel decoder:
        # (B, nq, C) @ (B, C, H_mask*W_mask) -> (B, nq, H_mask*W_mask)
        # mask_features shape: [B, C, H_mask, W_mask]
        mask_preds = torch.einsum("bqc,bchw->bqhw", query_mask_embeds, mask_features) # Output shape: [B, num_queries, H_mask, W_mask]

        return {"pred_logits": class_logits, "pred_masks": mask_preds}
