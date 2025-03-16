import torch
import torch.nn as nn
import torch.functional as F
import time
import random
import numpy as np


def set_seed(seed=42, deterministic_cudnn=True):
    random.seed(seed)
    # NumPy
    np.random.seed(seed) 
    # PyTorch (both CPU and CUDA)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    # Optional: Make CuDNN deterministic (might slow down training)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    


class CharacterTokenizer:
    def __init__(self, max_len=128):
        # Define a set of characters to tokenize (includes lowercase letters, digits, and common punctuation)
        chars = "abcdefghijklmnopqrstuvwxyz0123456789 .,;:!?'\"-()<>[]{}/"
        # Create mapping dictionaries
        self.char2idx = {char: idx for idx, char in enumerate(chars)}
        self.idx2char = {idx: char for idx, char in enumerate(chars)}
        
        # Add a special tokens for unknown characters, EOS and BOS
        self.unk_idx = len(chars)
        self.char2idx['<UNK>'] = self.unk_idx
        self.idx2char[self.unk_idx] = '<UNK>'

        self.bos_idx = self.unk_idx + 1
        self.char2idx['<BOS>'] = self.bos_idx
        self.idx2char[self.bos_idx] = '<BOS>'
        
        self.eos_idx = self.bos_idx + 1
        self.char2idx['<EOS>'] = self.eos_idx
        self.idx2char[self.eos_idx] = '<EOS>'

        self.pad_idx = self.eos_idx + 1
        self.char2idx['<PAD>'] = self.pad_idx
        self.idx2char[self.pad_idx] = '<PAD>'
        
        
        # Vocabulary size
        self.vocab_size = len(self.char2idx)
        
        self.max_len=128
    
    def encode(self, text):
        """Convert a string to a list of token indices"""
        ids= [self.char2idx.get(char, self.unk_idx) for char in text.lower()][:self.max_len]
        return ids
        #if len(ids)<self.max_len:
        #    ids.extend([self.char2idx['<PAD>']]* (len(ids)-self.max_len))
    
    def decode(self, indices):
        """Convert a list of token indices back to a string"""
        return ''.join([self.idx2char.get(idx, '<UNK>') for idx in indices])
    
    def batch_encode(self, texts):
        """Encode a batch of texts"""
        return [self.encode(text) for text in texts]
    
    def batch_decode(self, batch_indices):
        """Decode a batch of token indices"""
        return [self.decode(indices) for indices in batch_indices]
    

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
        self.register_buffer("mask", torch.tril(torch.ones(T, T)).reshape(1, 1, T, T))

    def forward(self, x, cache=None, return_cache=False):
        # input shape is (B,T,d)
        B, T, d = x.shape
        
        qkv = self.kqv(x)  # shape of B,T,3*d
        q, k, v = qkv.chunk(3, dim=-1)  # each shape of B,T,d

        q = q.view(B, T, self.h, self.head_dim).transpose(1, 2)  # Shape of B,h,T,head_dim
        k = k.view(B, T, self.h, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.h, self.head_dim).transpose(1, 2)

        if cache is not None:
            old_k, old_v = cache
            k = torch.cat((old_k, k), dim=2)  # Append across T
            v = torch.cat((old_v, v), dim=2)
            S = k.size(2)  # Total sequence length (past + current)
        else:
            S = T

        attn_weights = (q @ k.transpose(-2, -1)) / torch.sqrt(self.head_dim)  # shape of B,h,T,T or B,h,T,S
        
        if cache is None:
            # Standard case - apply full causal mask
            attn_weights = attn_weights.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        else:
            # With cache - only need causal masking for multiple new tokens
            if T > 1:
                past_len = S - T
                # Apply causal mask to the portion where new tokens attend to new tokens
                causal_mask = self.mask[:, :, :T, :T] == 0
                attn_weights[:, :, :, past_len:] = attn_weights[:, :, :, past_len:].masked_fill(
                    causal_mask, float("-inf")
                )
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        scores = attn_weights @ v  # shape of B,h,T,head_dim
        
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
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
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
                'attn': causalMHA(T=max_txt_len, d=hidden_dims),
                'norm1': nn.LayerNorm(hidden_dims),
                'mlp': MLP(hidden_dims),
                'norm2': nn.LayerNorm(hidden_dims)
            }) for _ in range(n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_dims)
        self.output_proj = nn.Linear(hidden_dims, vocab_size)

    def forward(self, x):
        B, T = x.shape
        # embedding
        x = self.token_embedding(x) + self.pos_embedding[:, :T, :]
        # transformer layers
        for block in self.blocks:
            # attention block with residual connection and normalization
            residual = x
            x = residual + block['attn'](block['norm1'](x))        
            # mlp block with residual connection and normalization
            residual = x
            x = residual + block['mlp'](block['norm2'](x))
        
        # final normalization and projection
        x = self.final_norm(x)
        x = self.output_proj(x)
        return x



def generate_with_cache(model, prompt_tokens, num_tokens=50):
    x = torch.tensor([prompt_tokens]).to(model.device)  # Shape: [1, prompt_length]
    
    # Process the entire prompt and get initial output and cache
    start_time = time.time()
    output, kv_cache = model(x, return_cache=True)
    
    # Generate tokens one by one
    generated_tokens = []
    for i in range(num_tokens):
        # Get next token prediction
        next_token_logits = output[:, -1, :]  # Take logits of the last token
        # Greedy decoding
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)  # Shape: [batch_size, 1]
        generated_tokens.append(next_token.item())
        
        # Process only this new token with the cache
        output, kv_cache = model(next_token, cache=kv_cache, return_cache=True)
    
    end_time = time.time()
    
    return {
        "tokens": prompt_tokens + generated_tokens,
        "time_taken": end_time - start_time
    }


def generate_without_cache(model, prompt_tokens, num_tokens=50):
    # Start with the prompt tokens
    all_tokens = prompt_tokens.copy()
    x = torch.tensor([all_tokens]).to(model.device)
    
    start_time = time.time()
    
    # Generate tokens one by one
    for i in range(num_tokens):
        # Process the entire sequence so far
        output = model(x)
        
        # Get next token prediction
        next_token_logits = output[:, -1, :]  # Take logits of the last token
        # Greedy decoding
        next_token = torch.argmax(next_token_logits, dim=-1).item()
        
        # Append the new token to our sequence
        all_tokens.append(next_token)
        
        # Create a new input tensor with the updated sequence
        x = torch.tensor([all_tokens]).to(model.device)
    
    end_time = time.time()
    
    return {
        "tokens": all_tokens,
        "time_taken": end_time - start_time
    }


if __name__=="__main__": 

    inp_txt = "Sky is "
    tokenizer = CharacterTokenizer()
    input_ids= tokenizer.encode(inp_txt)
    inp_tensor = torch.tensor([input_ids])
    model=transformerDecoder()

    out_without_c=generate_without_cache(model,inp_tensor)
    out_with_c=generate_with_cache(model,inp_tensor)




        