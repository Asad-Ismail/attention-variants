import torch
import torch.nn as nn


class CharacterTokenizer:
    def __init__(self):
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
        
        # Vocabulary size
        self.vocab_size = len(self.char2idx)
    
    def encode(self, text):
        """Convert a string to a list of token indices"""
        return [self.char2idx.get(char, self.unk_idx) for char in text.lower()]
    
    def decode(self, indices):
        """Convert a list of token indices back to a string"""
        return ''.join([self.idx2char.get(idx, '<UNK>') for idx in indices])
    
    def batch_encode(self, texts):
        """Encode a batch of texts"""
        return [self.encode(text) for text in texts]
    
    def batch_decode(self, batch_indices):
        """Decode a batch of token indices"""
        return [self.decode(indices) for indices in batch_indices]
    


class causalSelfAttention(nn.Module):
    def __init__(self, T=128, d = 512, h=8):
        super().__init__()
        # T is seq length , d is dimension of embedding and h is number of heads
        assert d%h==0, "Dimensions should be divisible by number of heads"
        self.T = T
        self.d = d
        self.register_buffer("mask",torch.tril(torch.ones(T,T))). reshape(1,1,T,T)

    def forward(x):
        # input shape is (B,T,d)
        B,T,d= x.shape


if __name__=="__main__": 

    inp_txt = "Sky is "
    tokenizer = CharacterTokenizer()
    




        