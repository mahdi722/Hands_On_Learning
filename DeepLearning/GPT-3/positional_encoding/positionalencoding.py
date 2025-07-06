import torch.nn as nn
import torch
import math

class Embedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()                      
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, tokens):
        # Tokens.size = (batch_size, sequence_length) 
        x = self.embed(tokens) # -> (batch_size, sequence_length, d_model)
        x = x * (self.d_model ** 0.5) # In the embedding layers, we multiply those weights by sqrt(d_model)
        return x
    
class Positional_Encoding(nn.Module):
    def __init__(self, d_model: int, max_length: int):
        super().__init__()

        pos = torch.arange(0, max_length).unsqueeze(1)

        div = torch.arange(0, d_model, 2)
        div_term = torch.ones(size=(1, d_model//2)).squeeze(0) / (100 ** (2 * div / d_model))

        self.PE = torch.zeros(size=(max_length, d_model))
        self.PE[:, 0::2] = torch.sin(pos * div_term)
        self.PE[:, 1::2] = torch.cos(pos * div_term)
        # self.PE (1, max_len, d_model)


        
    def forward(self, tensor):
        # tensor = (batch_size, seq length, d_model)

        tensor = tensor + self.PE[:tensor.size(1), :] #(batch_size, seq length, d_model)

        return tensor

class EmbeddingsWithPosition(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=5000):
        super().__init__()
        self.token_emb = Embedding(d_model, vocab_size)
        self.pos_enc   = Positional_Encoding(d_model, max_len)

    def forward(self, tokens):
        x = self.token_emb(tokens)    # embed & scale
        x = self.pos_enc(x)           # add positional encodings
        return x                      # ready to feed into the Transformer :)