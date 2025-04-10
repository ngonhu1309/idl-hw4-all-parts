import torch.nn as nn
import torch
from typing import Tuple, Optional
from .sublayers import SelfAttentionLayer, FeedForwardLayer

'''
TODO: Implement this Module.

This file contains the encoder layer implementation used in transformer architectures:

SelfAttentionEncoderLayer: Used in encoder part of transformers
- Contains self-attention and feed-forward sublayers
- Unlike decoder, does not use causal masking (can attend to all positions)
- Used for tasks like encoding input sequences where bidirectional context is needed

The layer follows a Pre-LN (Layer Normalization) architecture where:
- Layer normalization is applied before each sublayer operation
- Residual connections wrap around each sublayer

Implementation Steps:
1. Initialize the required sublayers in __init__:
  - SelfAttentionLayer for self-attention (no causal mask needed)
  - FeedForwardLayer for position-wise processing

2. Implement the forward pass to:
  - Apply sublayers in the correct order
  - Pass appropriate padding masks (no causal mask needed)
  - Return both outputs and attention weights
'''

class SelfAttentionEncoderLayer(nn.Module):
    '''
    Pre-LN Encoder Layer with self-attention mechanism.
    Used in the encoder part of transformer architectures.
    '''
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        '''
        Initialize the SelfAttentionEncoderLayer. 
        Args:
            d_model  (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            d_ff    (int): The dimension of the feedforward network.
            dropout (float): The dropout rate.
        '''
        super().__init__()
        # Initialize the sublayers
        self.self_attn = SelfAttentionLayer(d_model, num_heads, dropout)
        self.ffn = FeedForwardLayer(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass for the EncoderLayer.
        Args:
            x (torch.Tensor): The input tensor. shape: (batch_size, seq_len, d_model)  
            key_padding_mask (torch.Tensor): The padding mask for the input. shape: (batch_size, seq_len)

        Returns:
            x (torch.Tensor): The output tensor. shape: (batch_size, seq_len, d_model)
            mha_attn_weights (torch.Tensor): The attention weights. shape: (batch_size, seq_len, seq_len)  
        '''
        # Self-attention sublayer
        residual = x
        x_norm = self.norm1(x)
        x_attn, mha_attn_weights = self.self_attn(x_norm, key_padding_mask=key_padding_mask)
        x = residual + x_attn

        # Feed-forward sublayer
        x_norm = self.norm2(x)
        x = self.ffn(x_norm)  # FFN output is the final output

        return x, mha_attn_weights
