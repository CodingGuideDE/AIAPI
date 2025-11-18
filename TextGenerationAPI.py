from fastapi import FastAPI
from model import Transformer
from model import TransformerConfig
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


app = FastAPI()

@app.get("/getText/{promt}/{temperature}")
def root(promt, temperature: float):
    transformer_config = TransformerConfig()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token 

    tokenizer.add_special_tokens({'bos_token': '<|bos|>'})
    tokenizer.add_special_tokens({'eos_token': '<|end|>'})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Transformer(transformerConfig=transformer_config).to(device)
    model.load_state_dict(torch.load("model_small.pth", map_location='cpu'))
    model.eval()

    input_text = promt
    input_ids = tokenizer.encode(input_text,return_tensors='pt').to(device)

    temperature = temperature


    with torch.no_grad():
        for i in range(transformer_config.max_len):
            
            logits = model(input_ids)

            logits = logits[:, -1, :] / temperature 

            probs = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)


            if tokenizer.decode([next_token.item()]) == "." :
                break

            input_ids = torch.cat([input_ids, next_token], dim=-1)
        generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print(generated_text)
    
        return {"Text": generated_text}
    
    return{"Fehler ": "Fehler in der Generation aufgetreten"}