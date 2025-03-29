import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        """
        Args:
            embed_dim (int): The embedding dimension.
            num_heads (int): The number of attention heads.
            dropout (float): Dropout probability on attention weights.
        """
        super(MultiCrossAttention, self).__init__()
        # Initialize the built-in multi-head attention module.
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        # Optionally, you can add a layer norm and a feed-forward network here to mimic a full transformer block.
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
    
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        Args:
            query (Tensor): Query tensor of shape (query_len, batch_size, embed_dim).
            key (Tensor): Key tensor of shape (key_len, batch_size, embed_dim).
            value (Tensor): Value tensor of shape (key_len, batch_size, embed_dim).
            attn_mask (Optional[Tensor]): Optional mask to prevent attention to certain positions.
            key_padding_mask (Optional[Tensor]): Optional mask to ignore padding in keys.
        
        Returns:
            Tensor: The output tensor after applying multi-head cross-attention and feed-forward network.
        """
        # Compute multi-head cross attention
        attn_output, attn_weights = self.multihead_attn(query, key, value,
                                                          attn_mask=attn_mask,
                                                          key_padding_mask=key_padding_mask)
        # Residual connection and layer normalization
        query = query + attn_output
        query = self.norm(query)
        
        # Optionally, pass through a feed-forward network (with another residual connection)
        ffn_output = self.ffn(query)
        output = self.norm(query + ffn_output)
        
        return output, attn_weights