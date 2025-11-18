import torch.nn as nn 
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import GPT2Tokenizer, T5Tokenizer
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import re

@dataclass
class TransformerConfig:
    d_model: int = 256
    n_heads: int = 8
    vocab_size: int = GPT2Tokenizer.from_pretrained('gpt2').vocab_size + 2
    max_len = 64
    dropout: float = 0.1
    num_epochs: int = 30


class MultiHeadAttention(nn.Module):
    def __init__(self, transformerConfig, is_causual):
        super(MultiHeadAttention, self).__init__()
        self.transformer_config = transformerConfig
        self.is_causual = is_causual

        self.attention_dropout = nn.Dropout(transformerConfig.dropout)
        self.residual_dropout = nn.Dropout(transformerConfig.dropout)

        self.q_layer = nn.Linear(transformerConfig.d_model, transformerConfig.d_model)
        self.k_layer = nn.Linear(transformerConfig.d_model, transformerConfig.d_model)
        self.v_layer = nn.Linear(transformerConfig.d_model, transformerConfig.d_model)


        self.c_proj = nn.Linear(transformerConfig.d_model, transformerConfig.d_model)


        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        if not self.flash:
            print("Flash Attention not available, using standard attention.")

    def forward(self, x, mask=None):
        B, T, _ = x.size()

        Q = self.q_layer(x)
        K = self.k_layer(x)
        V = self.v_layer(x)

        K = K.view(B, T, self.transformer_config.n_heads, self.transformer_config.d_model // self.transformer_config.n_heads).transpose(1, 2)
        V = V.view(B, T, self.transformer_config.n_heads, self.transformer_config.d_model // self.transformer_config.n_heads).transpose(1, 2)
        Q = Q.view(B, T, self.transformer_config.n_heads, self.transformer_config.d_model // self.transformer_config.n_heads).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=self.transformer_config.dropout, is_causal=self.is_causual)
        else:
            att = torch.matmul(Q, K.transpose(-2, -1)) / ((self.transformer_config.d_model // self.transformer_config.n_heads) ** 0.5)

            if mask is not None:
                att = att.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))

            if self.is_causual:
                mask = torch.tril(torch.ones(Q.size(-2), K.size(-2), device=Q.device)).unsqueeze(0).unsqueeze(0)
                att = att.masked_fill(mask == 0, float('-inf'))
                
            att = F.softmax(att, dim=-1)
            att = self.attention_dropout(att)
            y = torch.matmul(att, V)

        y = y.transpose(1, 2).contiguous().view(B, T, self.transformer_config.d_model)
        y = self.c_proj(y)
        y = self.residual_dropout(y)

        return y
    

class Decoder(nn.Module):
    def __init__(self, transformerConfig):
        super(Decoder, self).__init__()
        self.transformer_config = transformerConfig

        self.fc_out = nn.Linear(transformerConfig.d_model, transformerConfig.vocab_size)
        self.dropout = nn.Dropout(transformerConfig.dropout)

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(transformerConfig)
            for _ in range(transformerConfig.decoder_layer)
        ])

        self.token_embedding = nn.Embedding(transformerConfig.vocab_size, transformerConfig.d_model)
        self.position_embedding = nn.Embedding(transformerConfig.max_len, transformerConfig.d_model)

    def forward(self, x, mask=None):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        x = self.token_embedding(x.long()) + self.position_embedding(positions)
        x = self.dropout(x)

        for block in self.decoder_blocks:
            x = block(x, mask)

        logits = self.fc_out(x)
        return logits



class DecoderBlock(nn.Module):
    def __init__(self, transformerConfig):
        super(DecoderBlock, self).__init__()

        self.masked_layer_norm = nn.LayerNorm(transformerConfig.d_model)
        self.feed_forward_layer_norm = nn.LayerNorm(transformerConfig.d_model)

        self.masked_attention = MultiHeadAttention(transformerConfig, is_causual=True)

        self.feed_forward = nn.Sequential(
            nn.Linear(transformerConfig.d_model,  4 * transformerConfig.d_model),
            nn.Dropout(transformerConfig.dropout),
            nn.GELU(),
            nn.Linear(transformerConfig.d_model * 4, transformerConfig.d_model),
            nn.Dropout(transformerConfig.dropout)
        )

        self.fc_out = nn.Linear(transformerConfig.d_model, transformerConfig.vocab_size)
        self.dropout = nn.Dropout(transformerConfig.dropout)

    def forward(self, x, mask=None):

        att = self.masked_attention(x, mask)
        x = self.masked_layer_norm(x + att)
        x = self.dropout(x)

        ff = self.feed_forward(x)
        x = self.feed_forward_layer_norm(x + ff)

        return x

class Transformer(nn.Module):
    def __init__(self, transformerConfig):
        super(Transformer, self).__init__()
        self.transformer_config = transformerConfig
        self.layernorm = nn.LayerNorm(transformerConfig.d_model)
        self.decoder = Decoder(transformerConfig)

    def forward(self, x, mask = None):
        logits = self.decoder(x, mask)
        return logits