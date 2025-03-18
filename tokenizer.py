
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
    