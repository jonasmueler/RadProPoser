import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        """
        Sinusoidal positional encoding module.

        Args:
            embed_dim (int): Dimension of the embedding space.
            max_len (int): Maximum length of input sequences.
        """
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds positional encodings to the input.

        Args:
            x (Tensor): Input tensor of shape (seq_len, batch_size, embed_dim).

        Returns:
            Tensor: Input tensor with positional encodings added.
        """
        seq_len = x.size(0)
        return x + self.pe[:seq_len, :]

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, max_len=384, dropout=0.001):
        """
        Transformer Encoder with positional encodings.

        Args:
            embed_dim (int): Dimensionality of input embeddings.
            num_heads (int): Number of attention heads.
            ff_dim (int): Dimensionality of the feedforward network.
            num_layers (int): Number of encoder layers.
            max_len (int): Maximum length of input sequences.
            dropout (float): Dropout rate for regularization.
        """
        super(TransformerEncoder, self).__init__()
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(embed_dim, max_len)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=ff_dim, 
            dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, 
            num_layers=num_layers
        )

    def forward(self, src, mask=None, src_key_padding_mask=None):
        """
        Forward pass for the Transformer encoder.

        Args:
            src (Tensor): Input tensor of shape (seq_len, batch_size, embed_dim).
            mask (Tensor, optional): Mask for attention weights (seq_len, seq_len).
            src_key_padding_mask (Tensor, optional): Mask for padded positions (batch_size, seq_len).

        Returns:
            Tensor: Output tensor of shape (seq_len, batch_size, embed_dim).
        """
        # Add positional encoding
        src = self.positional_encoding(src)
        # Pass through Transformer encoder
        return self.encoder(src, mask=mask, src_key_padding_mask=src_key_padding_mask)
