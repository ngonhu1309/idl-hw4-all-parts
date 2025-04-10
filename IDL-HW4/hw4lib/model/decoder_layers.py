import torch.nn as nn
import torch
from typing import Tuple, Optional
from .sublayers import SelfAttentionLayer, CrossAttentionLayer, FeedForwardLayer

'''
TODO: Implement these Modules.

This file contains two key decoder layer implementations used in transformer architectures:

1. SelfAttentionDecoderLayer: Used in decoder-only transformers (like GPT)
   - Contains masked self-attention and feed-forward sublayers
   - Used for tasks like language modeling where only previous tokens can be attended to
   
2. CrossAttentionDecoderLayer: Used in encoder-decoder transformers (like BART)
   - Contains masked self-attention, cross-attention, and feed-forward sublayers
   - Used for tasks like translation where decoder needs to attend to encoder outputs

Each layer follows a Pre-LN (Layer Normalization) architecture where:
- Layer normalization is applied before each sublayer operation
- Residual connections wrap around each sublayer

Implementation Steps for Each Layer:
1. Initialize the required sublayers in __init__:
   - SelfAttentionLayer for masked self-attention
   - CrossAttentionLayer for cross-attention (in CrossAttentionDecoderLayer only)
   - FeedForwardLayer for position-wise processing

2. Implement the forward pass to:
   - Apply sublayers in the correct order
   - Pass appropriate masks to attention layers
   - Return both outputs and attention weights
'''

## -------------------------------------------------------------------------------------------------  
## Decoder Layers
## -------------------------------------------------------------------------------------------------      
class SelfAttentionDecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
       
        self.self_attn = SelfAttentionLayer(d_model, num_heads, dropout)
        self.ffn = FeedForwardLayer(d_model, d_ff, dropout)
        
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention sublayer
        residual = x
        norm_x1 = self.norm1(x)
        attn_output, attn_weights = self.self_attn(norm_x1, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = residual + self.dropout1(attn_output)
        
        # Feed-forward sublayer
        norm_x2 = self.norm2(x)
        x = self.ffn(norm_x2)  # FFN output becomes the final output
        
        return x, attn_weights


## -------------------------------------------------------------------------------------------------    
class CrossAttentionDecoderLayer(nn.Module):
    '''
    Pre-LN Decoder Layer with masked self-attention, cross-attention, and feed-forward sublayers.
    Used in the encoder-decoder Transformer architecture.
    '''
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        '''
        Initialize the CrossAttentionDecoderLayer. 
        Args:
            d_model   (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            d_ff      (int): The dimension of the feedforward network.
            dropout (float): The dropout rate.
        '''
        super().__init__()

        # Initialize the sublayers  
        self.self_attn = SelfAttentionLayer(d_model, num_heads, dropout)  # Masked self-attention layer
        self.cross_attn = CrossAttentionLayer(d_model, num_heads, dropout)  # Cross-attention layer
        self.ffn = FeedForwardLayer(d_model, d_ff, dropout)  # Feed-forward network

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, dec_key_padding_mask: Optional[torch.Tensor] = None, enc_key_padding_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Forward pass for the CrossAttentionDecoderLayer.
        Args:
            x (torch.Tensor): The input tensor. shape: (batch_size, seq_len, d_model)   
            enc_output (torch.Tensor): The encoder output. shape: (batch_size, seq_len, d_model)
            dec_key_padding_mask (Optional[torch.Tensor]): The padding mask for the decoder input. shape: (batch_size, seq_len)
            enc_key_padding_mask (Optional[torch.Tensor]): The padding mask for the encoder output. shape: (batch_size, seq_len')
            attn_mask (Optional[torch.Tensor]): The self-attention mask for the decoder input. shape: (seq_len, seq_len)
        Returns:
            x (torch.Tensor): The output tensor. shape: (batch_size, seq_len, d_model)
            self_attn_weights (torch.Tensor): The attention weights. shape: (batch_size, seq_len, seq_len)   
            cross_attn_weights (torch.Tensor): The attention weights. shape: (batch_size, seq_len, seq_len)    
        '''
        # Self-attention sublayer
        self_attn_output, self_attn_weights = self.self_attn(
            x=x,
            key_padding_mask=dec_key_padding_mask,
            attn_mask=attn_mask
        )

        # Cross-attention sublayer
        cross_attn_output, cross_attn_weights = self.cross_attn(
            x=self_attn_output,
            y=enc_output,
            key_padding_mask=enc_key_padding_mask,
            attn_mask=None
        )

        # Feed-forward sublayer
        output = self.ffn(cross_attn_output)

        return output, self_attn_weights, cross_attn_weights
## -------------------------------------------------------------------------------------------------    
