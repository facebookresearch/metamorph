
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, include_ffn=False):
        super().__init__()
        self.include_ffn = include_ffn
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        if self.include_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, 4 * embed_dim),
                nn.ReLU(),
                nn.Linear(4 * embed_dim, embed_dim)
            )
            self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, tokens, x_proj):
        # Cross-Attention
        attn_output, _ = self.cross_attn(query=tokens, key=x_proj, value=x_proj)
        tokens = self.norm1(tokens + attn_output)
        if self.include_ffn:
            # Feed-Forward Network
            ffn_output = self.ffn(tokens)
            tokens = self.norm2(tokens + ffn_output)
        return tokens

class SimplifiedSigLIPProjector(nn.Module):
    def __init__(self, input_dim=1152, hidden_dim=4096, output_dim=768, num_tokens=77, num_layers=6, num_heads=8, mode='mlp'):
        super().__init__()
        self.config = SimpleNamespace(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_tokens=num_tokens,
            num_layers=num_layers,
            num_heads=num_heads,
            mode=mode
        )

        self.mode = mode
        self.num_layers = num_layers
        
        if self.mode == 'mlp':
            self.layers = nn.ModuleList()
            self.norms = nn.ModuleList()
            
            # First layer
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
            
            # Middle layers
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.norms.append(nn.LayerNorm(hidden_dim))
            
            # Last layer
            self.layers.append(nn.Linear(hidden_dim, output_dim))
            self.norms.append(nn.LayerNorm(output_dim))
            
        elif self.mode in ['xattn', 'xattnffn']:
            self.num_tokens = num_tokens
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.num_heads = num_heads
            # Learnable token embeddings
            self.token_embeddings = nn.Parameter(torch.randn(1, num_tokens, output_dim))
            # Input projection to match output_dim
            self.proj = nn.Linear(input_dim, output_dim)
            self.input_norm = nn.LayerNorm(output_dim)
            # Cross-Attention Blocks
            include_ffn = (self.mode == 'xattnffn')
            self.cross_attn_layers = nn.ModuleList([
                CrossAttentionBlock(embed_dim=output_dim, num_heads=num_heads, include_ffn=include_ffn)
                for _ in range(num_layers)
            ])
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
    @classmethod
    def from_config(cls, config):
        return cls(**vars(config))
    
    def forward(self, x):
        if self.mode == 'mlp':
            for layer, norm in zip(self.layers[:-1], self.norms[:-1]):
                x = F.relu(norm(layer(x)))
            return self.norms[-1](self.layers[-1](x))
        elif self.mode in ['xattn', 'xattnffn']:
            # x: (batch_size, seq_len, input_dim)
            batch_size = x.size(0)
            # Project input to output_dim
            x_proj = self.proj(x)  # (batch_size, seq_len, output_dim)
            x_proj = self.input_norm(x_proj)
            # Initialize tokens
            tokens = self.token_embeddings.expand(batch_size, -1, -1)  # (batch_size, num_tokens, output_dim)
            # Pass through Cross-Attention Blocks
            for layer in self.cross_attn_layers:
                tokens = layer(tokens, x_proj)
            return tokens  # (batch_size, num_tokens, output_dim)

