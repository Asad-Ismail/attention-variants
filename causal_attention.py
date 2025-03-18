import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalMHA(nn.Module):
    def __init__(self, T=128, d=512, h=8):
        super().__init__()
        # T is seq length, d is dimension of embedding and h is number of heads
        assert d % h == 0, "Dimensions should be divisible by number of heads"
        self.T = T
        self.d = d
        self.h = h
        self.head_dim = self.d // self.h
        self.kqv = nn.Linear(d, 3*d)
        self.proj = nn.Linear(d, d)

    def forward(self, x, cache=None, return_cache=False):
        # input shape is (B,T,d)
        B, T, d = x.shape
        
        qkv = self.kqv(x)  # shape of B,T,3*d
        q, k, v = qkv.chunk(3, dim=-1)  # each shape of B,T,d

        q = q.view(B, T, self.h, self.head_dim).transpose(1, 2)  # Shape of B,h,T,head_dim
        k = k.view(B, T, self.h, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.h, self.head_dim).transpose(1, 2)

        if cache is not None:
            past_k, past_v = cache
            # Concatenate past and present keys and values
            k = torch.cat((past_k, k), dim=2)  # Append across T
            v = torch.cat((past_v, v), dim=2)
        
        # Calculate attention scores
        attn_weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # shape: B,h,T,S

        with torch.no_grad():
            S = k.size(2)  # Total sequence length with cache
            causal_mask = torch.triu(torch.ones(T, S, device=x.device), diagonal=1).bool()
            
            # If using cache, we need to adjust the mask
            if cache is not None:
                past_length = past_k.size(2)
                # Allow each query to attend to all past cached tokens
                causal_mask[:, :past_length] = False
        
        # maskfill will equate/replace value where mask is True
        attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        
        # Continue with attention computation
        attn_weights = F.softmax(attn_weights, dim=-1)
        scores = attn_weights @ v  # shape of B,h,T,S
        
        scores = scores.transpose(1, 2).contiguous().view(B, T, d)  # B,T,d
        out = self.proj(scores)  # B,T,d

        if return_cache:
            new_cache = (k, v)
            return out, new_cache
        return out
    


class MLP(nn.Module):
    def __init__(
        self,
        d,
        bias=False,
        dropout=0.2
    ):
        """
        Arguments:
        d: size of embedding dimension
        bias: whether or not to use bias in linear layers
        dropout: probability of dropout
        """
        super().__init__()
        self.c_fc    = nn.Linear(d, 4 * d, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * d, d, bias=bias)
        #self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        #x = self.dropout(x)
        return x


class transformerDecoder(nn.Module):

    def __init__(self, vocab_size=58, hidden_dims=512, max_txt_len=128, n_layers=1):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dims = hidden_dims
        
        # token embedding and positional encoding
        self.token_embedding = nn.Embedding(vocab_size, hidden_dims)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_txt_len, hidden_dims))

        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'attn': CausalMHA(T=max_txt_len, d=hidden_dims),
                'norm1': nn.LayerNorm(hidden_dims),
                'mlp': MLP(hidden_dims),
                'norm2': nn.LayerNorm(hidden_dims)
            }) for _ in range(n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_dims)
        self.output_proj = nn.Linear(hidden_dims, vocab_size)

    @property
    def device(self):
        return next(self.parameters()).device
    

    def forward(self, x, cache=None, return_cache=False):
        B, T = x.shape
        # Initialize cache and create embeddings 
        if cache is None:
            cache = [None] * self.n_layers
            x = self.token_embedding(x) + self.pos_embedding[:, :T, :]  
        else:
            S = cache[0][0].size(2) + T  # Total sequence length is cached length + current input length
            x = self.token_embedding(x) + self.pos_embedding[:, S - T : S, :] 
        # Transformer layers
        new_cache = []
        for i, block in enumerate(self.blocks):
            # Attention block with residual connection and normalization
            residual = x
            x, layer_cache = block['attn'](
                block['norm1'](x),
                cache=cache[i],
                return_cache=True  # Always return cache for each layer
            )
            x = residual + x
            new_cache.append(layer_cache)  # Store updated cache for this layer
            
            # MLP block with residual connection and normalization
            residual = x
            x = residual + block['mlp'](block['norm2'](x))
        
        # Final normalization and projection
        x = self.final_norm(x)
        x = self.output_proj(x)
        # Return output and optionally the updated cache
        if return_cache:
            return x, new_cache
        return x
