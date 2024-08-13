import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import pytorch_lightning as pl
from typing import List
import tqdm
import os
import csv 

class RelativePosition(nn.Module):

    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * self.d_model - 1, self.nhead)))
        nn.init.xavier_uniform_(self.relative_position_bias_table)

    def forward(self, length_query: int, length_key: int) -> torch.Tensor:
        indices_query = torch.arange(length_query, device=self.relative_position_bias_table.device)
        indices_key = torch.arange(length_key, device=self.relative_position_bias_table.device)
        distance_matrix = indices_key.unsqueeze(0) - indices_query.unsqueeze(1)
        distance_matrix_clipped = torch.clamp(distance_matrix, -(self.d_model-1), self.d_model-1)
        final_matrix = distance_matrix_clipped + self.d_model - 1
        embeddings = self.relative_position_bias_table[final_matrix.to(torch.long)]
        return embeddings
    


class AttentionHead(nn.Module):

    def __init__(self, hidden_size, d_model, k_bias_matrix, v_bias_matrix):
        super().__init__()
        self.d_model = d_model
        self.query_weights: nn.Linear = nn.Linear(hidden_size, self.d_model)
        self.key_weights: nn.Linear = nn.Linear(hidden_size, self.d_model)
        self.value_weights: nn.Linear = nn.Linear(hidden_size, self.d_model)
        self.k_bias_matrix = k_bias_matrix
        self.v_bias_matrix = v_bias_matrix

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        query: torch.Tensor = self.query_weights(query) # (b_s, n_t, head_dim)
        key: torch.Tensor = self.key_weights(key) # (b_s, n_t, head_dim)
        value: torch.Tensor = self.value_weights(value) # (b_s, n_t, head_dim)
        # Self-Attention scores
        attn_1: torch.Tensor = torch.matmul(query, key.transpose(1, 2)) # Q*K^T:(b_s, n_t, n_t)
        # Relative Position Attention scores
        attn_2: torch.Tensor = torch.matmul(query.permute(1, 0, 2), self.k_bias_matrix.transpose(1, 2)).transpose(0, 1) # Q*K_shifting^T:(b_s, n_t, n_t)
        # Relation-aware Self-Attention scores
        att_scores: torch.Tensor = (attn_1 + attn_2)/(self.d_model ** 0.5)
        if mask is not None:
            mask = mask.to(torch.int)
            att_scores: torch.Tensor = att_scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        att_weights: torch.Tensor = F.softmax(att_scores, dim=-1)
        # Weighted sum of values
        values_1: torch.Tensor = torch.matmul(att_weights, value) # (b_s, n_t, head_dim)
        # Relative Position Representation for values
        values_2: torch.Tensor = torch.matmul(att_weights.permute(1, 0, 2), self.v_bias_matrix).transpose(0, 1) # (b_s, n_t, head_dim)
        # Relation-aware values
        n_value  = values_1 + values_2
        return n_value


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size = 1024, num_heads = 16, k = 64, seq_len = 1500):
        super().__init__()
        self.hidden_size: int = hidden_size
        self.num_heads: int = num_heads
        self.head_dim: int = hidden_size // num_heads
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.relative_position_k: torch.Tensor = RelativePosition(self.head_dim, k)
        self.relative_position_v: torch.Tensor = RelativePosition(self.head_dim, k)
        self.k_bias_matrix: torch.Tensor = self.relative_position_k(seq_len, seq_len)
        self.v_bias_matrix: torch.Tensor = self.relative_position_v(seq_len, seq_len)
        self.attention_heads: nn.ModuleList = nn.ModuleList([AttentionHead(self.hidden_size, self.head_dim, self.k_bias_matrix, self.v_bias_matrix) for _ in range(self.num_heads)])
        self.fc: nn.Linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        attention_outputs: List[torch.Tensor] = [attention_head(query, key, value, mask=mask) for attention_head in self.attention_heads]
        hidden_state: torch.Tensor = torch.cat(attention_outputs, dim=-1)
        hidden_state: torch.Tensor = self.fc(hidden_state)
        return hidden_state

class DecoderLayer(nn.Module):
    def __init__(self, d_model=1024, num_heads=16, dim_feedforward=4096, dropout=0.1, k=64, seq_len=1500):
        super(DecoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, k, seq_len)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-head attention
        attn_output = self.attention(x, x, x)
        x = self.layer_norm1(x + self.dropout(attn_output))
        # Feed-forward network
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))
        return x
    
class Decoder(pl.LightningModule):
    def __init__(self, d_model=1024, num_layers = 12, num_heads=16, dim_feedforward=4096, dropout=0.1, k=64, seq_len=1500, vocab_size = 1500):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, dim_feedforward, dropout, k, seq_len)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = self.configure_optimizers()

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        output = self.linear(self.norm(x))    
        return output
    
    def training_step(self, batch, batch_idx):
        input_ids, target_ids = batch
        output = self(input_ids)
        output = output.view(-1, output.size(-1))  # (batch_size * seq_len, vocab_size)
        target_ids = target_ids.view(-1)  # (batch_size * seq_len)
        loss = self.loss_fn(output, target_ids)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-4)  # Replace with your preferred optimizer
        return optimizer
    
    def train(self, train_dataset, first_training = True, num_epochs = 10, checkpoint_dir = "./checkpoints", checkpoints_interval = 1, loss_file = "losses.csv", steps_per_checkpoint = 50):
        os.makedirs(checkpoint_dir, exist_ok=True) 
        loss_values = []

        # Initialize loss file
        with open(loss_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Step', 'Loss'])

        step_counter = 0  # To keep track of steps for loss file

        for epoch in range(num_epochs):
            epoch_loss = 0
            
            for input_ids, target_ids in tqdm.tqdm(train_dataset):
                self.optimizer.zero_grad()

                # Forward pass
                output = self(input_ids)
                output = output.view(-1, output.size(-1))
                target_ids = target_ids.view(-1)
                
                # Compute loss
                loss = self.loss_fn(output, target_ids)
                epoch_loss += loss.item()
                
                # Save loss value to file
                with open(loss_file, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([step_counter + 1, loss.item()])
                step_counter += 1
                
                # Backward pass
                loss.backward(retain_graph = True)
                self.optimizer.step()

                if step_counter % steps_per_checkpoint == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{epoch+1}_{step_counter}.pt')
                    torch.save({
                        'epoch': epoch + 1,
                        'step': step_counter,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': epoch_loss / len(train_dataset),
                    }, checkpoint_path)
                    print(f"Checkpoint saved at {checkpoint_path}")

            # Print or log epoch loss
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_dataset)}")
            
            # Save checkpoint
            if (epoch + 1) % checkpoints_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': epoch_loss / len(train_dataset),
                }, checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")


    
'''
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
'''