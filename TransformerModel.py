import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import pytorch_lightning as pl

class PositionEncoding(nn.Module):
    
    def __init__(self, d_model=1500, max_len=1500):

        super().__init__()

        pe = torch.zeros(max_len, d_model)
    
        position = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1)

        embedding_index = torch.arange(start=0, end=d_model, step=2).float()

        div_term = 1/torch.tensor(10000.0)**(embedding_index / d_model)
        
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 

        self.register_buffer('pe', pe) 

    def forward(self, tokens):
    
        return tokens + self.pe[:tokens.size(1), :]

class Attention(nn.Module): 
    
    def __init__(self, d_model=1500):
        
        super().__init__()
        
        self.d_model=d_model

        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        
        self.row_dim = 1
        self.col_dim = 2

        
    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask=None):

        q = self.W_q(encodings_for_q)
        k = self.W_k(encodings_for_k)
        v = self.W_v(encodings_for_v)

        sims = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1=self.col_dim))

        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5)

        if mask is not None:

            scaled_sims = scaled_sims.masked_fill(mask=mask, value=-1e9) 

        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)

        attention_scores = torch.matmul(attention_percents, v)
        
        return attention_scores

class DecoderOnlyTransformer(pl.LightningModule):
    
    def __init__(self, num_tokens=1500, d_model=1500, max_len=14000):
        
        super().__init__()
    
        
        pl.seed_everything(seed=42)
        self.num_tokens = num_tokens
        self.we = nn.Embedding(num_embeddings=num_tokens, 
                               embedding_dim=d_model)     
        
        self.pe = PositionEncoding(d_model=d_model, 
                                   max_len=max_len)

        self.self_attention = Attention(d_model=d_model)

        self.fc_layer = nn.Linear(in_features=d_model, out_features=num_tokens)
        
        self.loss = nn.CrossEntropyLoss()
        
        
    def forward(self, token_ids):
                
        word_embeddings = self.we(token_ids)        
        position_encoded = self.pe(word_embeddings)
        
        mask = torch.tril(torch.ones((token_ids.size(1), token_ids.size(1)), device=self.device))

        mask = mask == 0
        
        self_attention_values = self.self_attention(position_encoded, 
                                                    position_encoded, 
                                                    position_encoded, 
                                                    mask=mask)
                
        residual_connection_values = position_encoded + self_attention_values
        
        fc_layer_output = self.fc_layer(residual_connection_values)
        
        return fc_layer_output
    
    
    def configure_optimizers(self): 

        return Adam(self.parameters(), lr=0.001)
    
    
    def training_step(self, batch, batch_idx): 
        input_tokens, labels = batch
        output = self(input_tokens)
        loss = self.loss(output.view(-1, output.size(-1)), labels.view(-1))
        return loss
