import torch
from torch import nn
from cs336_basics.model.Embedding import Embedding
from cs336_basics.model.TransformerBlock import TransformerBlock
from cs336_basics.model.RMSNorm import RMSNorm
from cs336_basics.model.Linear import Linear
from cs336_basics.nn_utils import softmax


class TransformerLM(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta, 
                 vocab_size, context_length, num_layers):
        super().__init__()
        self.embeddings = Embedding(vocab_size, d_model)
        self.transformer_layers = nn.ModuleList()
        for i in range(num_layers):
            self.transformer_layers.append(TransformerBlock(d_model, num_heads, d_ff, context_length, theta))
        self.norm = RMSNorm(d_model=d_model)
        self.output_embedding = Linear(d_model, vocab_size)
        
    
    def forward(self, in_indices):
        in_features = self.embeddings(in_indices) # shape becomes becomes b, s, d_model
        for transformer_block in self.transformer_layers:
            in_features = transformer_block(in_features)
        
        norm_out = self.norm(in_features)
        output_embed = self.output_embedding(norm_out)
        return output_embed

        