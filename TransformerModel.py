import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import pytorch_lightning as pl
from typing import List
import tqdm
import os
import csv 
from torchviz import make_dot

'''
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
'''
class RelativePosition(nn.Module):

    def __init__(self, d_a: int, k: int):
        """
        Initialize the RelativePosition module.

        Args:
        - d_a (int): Number of dimensions in the relative position embeddings.
        - k (int): Clipping distance.
        """
        super().__init__()
        self.d_a = d_a
        self.k = k
        self.position_embeddings = nn.Parameter(torch.empty((2 * k + 1, d_a)))
        nn.init.xavier_uniform_(self.position_embeddings)

    def forward(self, length_query: int, length_key: int) -> torch.Tensor:
        """
        Compute relative position embeddings.

        Args:
        - length_query (int): Length of the query sequence.
        - length_key (int): Length of the key sequence.

        Returns:
        - embeddings (torch.Tensor): Relative position embeddings (length_query, length_key, embedding_dim).
        """
        # Generate relative position embeddings
        indices_query = torch.arange(length_query, device=self.position_embeddings.device)
        indices_key = torch.arange(length_key, device=self.position_embeddings.device)
        distance_matrix = indices_key.unsqueeze(0) - indices_query.unsqueeze(1)
        distance_matrix_clipped = torch.clamp(distance_matrix, -self.k, self.k)
        final_matrix = distance_matrix_clipped + self.k
        embeddings = self.position_embeddings[final_matrix.to(torch.long)]

        return embeddings



class AttentionHead(nn.Module):

    def __init__(self, hidden_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.query_weights: nn.Linear = nn.Linear(hidden_size, self.d_model)
        self.key_weights: nn.Linear = nn.Linear(hidden_size, self.d_model)
        self.value_weights: nn.Linear = nn.Linear(hidden_size, self.d_model)
        
    def forward(self, queryIn: torch.Tensor, keyIn: torch.Tensor, valueIn: torch.Tensor, maskIn: torch.Tensor = None, k_bias_matrix: torch.Tensor = None, v_bias_matrix: torch.Tensor = None) -> torch.Tensor:

        #self.k_bias_matrix = torch.randn_like(self.k_bias_matrix)
        #self.v_bias_matrix = torch.randn_like(self.v_bias_matrix)
       
        query: torch.Tensor = self.query_weights(queryIn) # (b_s, n_t, head_dim)
        key: torch.Tensor = self.key_weights(keyIn) # (b_s, n_t, head_dim)
        value: torch.Tensor = self.value_weights(valueIn) # (b_s, n_t, head_dim)
        # Self-Attention scores
        attn_1: torch.Tensor = torch.matmul(query, key.transpose(1, 2)) # Q*K^T:(b_s, n_t, n_t)

        if k_bias_matrix is None:
            # Create a zero tensor with the appropriate shape if k_bias_matrix is not provided
            k_bias_matrix = torch.zeros_like(attn_1, device=query.device)
            
        # Relative Position Attention scoresprint(
        attn_2: torch.Tensor = torch.matmul(query.transpose(0, 1), k_bias_matrix.transpose(1, 2)).transpose(0, 1) # Q*K_shifting^T:(b_s, n_t, n_t)
        # Relation-aware Self-Attention scores
        att_scores: torch.Tensor = (attn_1 + attn_2)/(self.d_model ** 0.5)
        if maskIn is not None:
            mask = maskIn.to(torch.int)
            att_scores_filled: torch.Tensor = att_scores.masked_fill(mask == 0, -1e9)
        else:
            att_scores_filled: torch.Tensor = att_scores
        att_weights: torch.Tensor = F.softmax(att_scores_filled, dim=-1)
        # Weighted sum of values
        values_1: torch.Tensor = torch.matmul(att_weights, value) # (b_s, n_t, head_dim)
        
        # Relative Position Representation for values
        if v_bias_matrix is None:
            # Create a zero tensor with the appropriate shape if v_bias_matrix is not provided
            v_bias_matrix = torch.zeros_like(values_1, device=query.device)
            
        values_2: torch.Tensor = torch.matmul(att_weights.transpose(0, 1), v_bias_matrix).transpose(0, 1) # (b_s, n_t, head_dim)
        # Relation-aware values
        n_value  = values_1 + values_2
        return n_value


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size =1024, num_heads=16, k=64, seq_len=1500):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size: int = hidden_size
        self.num_heads: int = num_heads
        self.head_dim: int = hidden_size // num_heads
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        # Learnable relative position embeddings
        self.relative_position_k = RelativePosition(self.head_dim, k)
        self.relative_position_v = RelativePosition(self.head_dim, k)

        # Multi-head attention layers
        self.attention_heads = nn.ModuleList([AttentionHead(self.hidden_size, self.head_dim) for _ in range(self.num_heads)])
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Recompute bias matrices using learned position embeddings
        k_bias_matrix = self.relative_position_k(self.seq_len, self.seq_len)  # These are derived from learnable embeddings
        v_bias_matrix = self.relative_position_v(self.seq_len, self.seq_len)

        # Pass through each attention head
        attention_outputs: List[torch.Tensor] = [attention_head(query, key, value, mask, k_bias_matrix, v_bias_matrix)
                                                 for attention_head in self.attention_heads]

        hidden_state: torch.Tensor = torch.cat(attention_outputs, dim=-1)
        hidden_state_final: torch.Tensor = self.fc(hidden_state)

        return hidden_state_final

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

    def forward(self, x, mask: torch.Tensor = None):
        
        # Multi-head attention
        attn_output = self.attention(x, x, x, mask)
        x_attended = self.layer_norm1(x + self.dropout(attn_output))
        # Feed-forward network
        ff_output = self.feed_forward(x_attended)
        x_final = self.layer_norm2(x_attended + self.dropout(ff_output))
        return x_final
    
class Decoder(pl.LightningModule):
    def __init__(self, d_model=1024, num_layers = 12, num_heads=16, dim_feedforward=4096, dropout=0.1, k=64, seq_len=1500, vocab_size = 1500, learning_rate = 10e-4):
        super().__init__()
        self.seq_len = seq_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, dim_feedforward, dropout, k, seq_len)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size)
        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = self.configure_optimizers()    
        self.automatic_optimization = False
        self.causal_mask = torch.tril(torch.ones((seq_len, seq_len))).unsqueeze(0).detach()
        self.causal_mask.requires_grad = False

    def forward(self, x, mask: torch.Tensor = None):
        
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        output = self.linear(self.norm(x))    
        return output
    

    def common_step(self, batch):
        input_ids, target_ids = batch

        with torch.autograd.set_detect_anomaly(True):
            output = self(input_ids, self.causal_mask)  # output shape: (batch_size, seq_len, vocab_size)
        
        batch_size, input_seq_len = input_ids.shape
        target_seq_len = target_ids.shape[1]
        
        output_target = output[:, input_seq_len - target_seq_len:, :]  # I take only the tokens I need to check (for the coarse generation I take only the coarse generated)
        
        # Reshape for CrossEntropyLoss
        output_target_reshaped = output_target.reshape(-1, output_target.size(-1))  # (batch_size * target_seq_len, vocab_size)
        target_ids_reshaped= target_ids.reshape(-1) 

        # Compute loss
        loss = self.loss_fn(output_target_reshaped, target_ids_reshaped)

        
        return loss, output_target_reshaped, target_ids_reshaped
        
    def training_step(self, batch, batch_idx):

        self.optimizer.zero_grad()

        loss, _, _ = self.common_step(batch)
        
        self.log('train_loss', loss)
        
        with torch.autograd.set_detect_anomaly(True):
            self.manual_backward(loss, retain_graph=False)

        self.optimizer.step()
        

        return loss
        
    def validation_step(self, batch, batch_idx):
        
        loss, output_target, target_ids = self.common_step(batch)

        labels_hat = torch.argmax(output_target, dim=1)
        val_acc = torch.sum(target_ids == labels_hat).item() / (len(target_ids) * 1.0)

        self.log_dict({'val_loss': loss, 'val_acc': val_acc})
        return {'val_loss': loss, 'val_acc': val_acc}

    def on_train_epoch_end(self):
        torch.save(self.state_dict(), f"checkpoints/{self.__class__.__name__}_epoch={self.current_epoch:02d}.ckpt")
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, weight_decay = 1e-4)  # Replace with your preferred optimizer
        return optimizer

    def generate_tokens(self, input_ids, padding_mask = None, num_tokens=10):

        generated_sequence = input_ids.unsqueeze(0)
        total_sequence = generated_sequence
        if padding_mask != None:
            padding_mask = padding_mask.unsqueeze(1)
        for _ in tqdm.tqdm(range(num_tokens)):
            with torch.no_grad():

                if padding_mask is not None:
                    # Combine causal mask and padding mask
                    combined_mask = self.causal_mask * padding_mask
                    padding_mask = torch.cat([padding_mask[:, :, 1:], torch.tensor([[[1]]])], dim = 2)
                    
                else:
                    combined_mask = self.causal_mask

                output = self(generated_sequence, combined_mask)  # output shape: [1, seq_len, vocab_size]
                next_token_logits = output[:, -1, :]  # logits for the last token
                probs = F.softmax(next_token_logits, dim=-1)  # convert to probabilities
                next_token = torch.argmax(probs, dim=-1)  # get the most probable token

                total_sequence = torch.cat((total_sequence, next_token.unsqueeze(0)), dim=1)
                generated_sequence = torch.cat((generated_sequence[:, 1:], next_token.unsqueeze(0)), dim=1)

        return total_sequence
    
    def pad_sequence(self, sequence, lenght):
        to_pad = lenght - sequence.shape[0]
        if to_pad > 0:
            padded_sequence = F.pad(sequence, (0, to_pad))
            padding_mask = torch.ones(1, lenght)
            padding_mask[0, sequence.shape[0]:] = 0
            return padded_sequence, padding_mask
        elif to_pad == 0:
            return sequence, None
        else:
            raise ValueError(f"Invalid token lenght, shorten the input to match this size {lenght}")

class SemanticTransformer(Decoder):
    def __init__(self, d_model=1024, num_layers = 12, num_heads=16, dim_feedforward=4096, dropout=0.1, k=64, audioDuration = 30, vocab_size = 500, learning_rate = 10e-4):
        
        self.seq_len = int(50 * audioDuration - 2)
        super().__init__(d_model, num_layers, num_heads, dim_feedforward, dropout, k, self.seq_len, vocab_size)

    def generate_tokens(self, semantic_tokens, num_tokens = 10):
        padded_semantic_tokens, padding_mask = self.pad_sequence(semantic_tokens, self.seq_len)
        return super().generate_tokens(padded_semantic_tokens, padding_mask, num_tokens)

    # Override of Decoder.common_step
    def common_step(self, batch):
        
        input_ids, target_ids = batch

        combined_mask = self.causal_mask * (input_ids != 0).float().unsqueeze(1).expand(-1, self.seq_len, self.seq_len)

        output = self(input_ids, combined_mask)  # output shape: (batch_size, seq_len, vocab_size)
        
        batch_size, input_seq_len = input_ids.shape
        target_seq_len = target_ids.shape[1]
        
        output_target = output[:, input_seq_len - target_seq_len:, :]  # I take only the tokens I need to check (for the coarse generation I take only the coarse generated)
        
        # Reshape for CrossEntropyLoss
        output_target = output_target.reshape(-1, output_target.size(-1))  # (batch_size * target_seq_len, vocab_size)
        target_ids= target_ids.reshape(-1) 

        # Compute loss
        loss = self.loss_fn(output_target, target_ids)
        return loss, output_target, target_ids

        
class CoarseTransformer(Decoder):
    def __init__(self, d_model=1024, num_layers = 12, num_heads=16, dim_feedforward=4096, dropout=0.1, k=64, audioDuration = 10, Q_prime = 3, vocab_size = 3072, learning_rate = 10e-4):

        self.semantic_size = int(50 * audioDuration - 1)
        self.coarse_size = int((50 * audioDuration + 1) * Q_prime) - 1

        self.seq_len = self.semantic_size + self.coarse_size
        
        super().__init__(d_model, num_layers, num_heads, dim_feedforward, dropout, k, self.seq_len, vocab_size, learning_rate)
        

    def generate_tokens(self, semantic_tokens, coarse_tokens, num_tokens = 10):
        padded_semantic_tokens, semantic_padding = self.pad_sequence(semantic_tokens, self.semantic_size)
        padded_coarse_tokens, coarse_padding = self.pad_sequence(coarse_tokens, self.coarse_size)
        sequence = torch.cat((padded_semantic_tokens, padded_coarse_tokens), dim=0)

        if semantic_padding is not None and coarse_padding is not None:
            padding_mask = torch.cat((semantic_padding, coarse_padding), dim=1)
        elif semantic_padding is not None:
            padding_needed = self.seq_len - semantic_padding.shape[1]
            padding_mask = torch.cat([semantic_padding, torch.ones(1, padding_needed)], dim=1)
        elif coarse_padding is not None:
            padding_needed = self.seq_len - coarse_padding.shape[1]
            padding_mask = torch.cat([torch.ones(1, padding_needed), coarse_padding], dim=1)
        else:
            padding_mask = None

        return super().generate_tokens(sequence, padding_mask, num_tokens)

    # Override of Decoder.common_step
    def common_step(self, batch):
        
        input_ids, target_ids = batch

        combined_mask = self.causal_mask * (input_ids != 0).float().unsqueeze(1).expand(-1, self.seq_len, self.seq_len)

        output = self(input_ids, combined_mask)  # output shape: (batch_size, seq_len, vocab_size)
        
        batch_size, input_seq_len = input_ids.shape
        target_seq_len = target_ids.shape[1]
        
        output_target = output[:, input_seq_len - target_seq_len:, :]  # I take only the tokens I need to check (for the coarse generation I take only the coarse generated)
        
        # Reshape for CrossEntropyLoss
        output_target = output_target.reshape(-1, output_target.size(-1))  # (batch_size * target_seq_len, vocab_size)
        target_ids= target_ids.reshape(-1) 

        # Compute loss
        loss = self.loss_fn(output_target, target_ids)
        return loss, output_target, target_ids

    

class FineTransformer(Decoder):
    def __init__(self, d_model=1024, num_layers = 12, num_heads=16, dim_feedforward=4096, dropout=0.1, k=64, audioDuration = 3, Q_prime = 3, Q = 8, vocab_size = 8192, learning_rate = 10e-4):

        self.coarse_size = int((50 * audioDuration + 1) * Q_prime)
        self.fine_size = int((50 * audioDuration + 1) * (Q - Q_prime) - 1)
        self.seq_len = self.coarse_size + self.fine_size 
        
        super().__init__(d_model, num_layers, num_heads, dim_feedforward, dropout, k, self.seq_len, vocab_size)
        

    def generate_tokens(self, coarse_tokens, fine_tokens, num_tokens = 10):
        padded_coarse_tokens, coarse_padding = self.pad_sequence(coarse_tokens, self.coarse_size)
        padded_fine_tokens, fine_padding = self.pad_sequence(fine_tokens, self.fine_size)
        sequence = torch.cat((padded_fine_tokens, padded_coarse_tokens), dim=0)

        if fine_padding != None and coarse_padding != None:
            padding_mask = torch.cat((coarse_padding, fine_padding), dim=1)
        elif coarse_padding != None:
            padding_needed = self.seq_len - coarse_padding.shape[1]
            padding_mask = torch.cat([coarse_padding, torch.ones(1, padding_needed)], dim=1)
        elif fine_padding != None:
            padding_needed = self.seq_len - fine_padding.shape[1]
            padding_mask = torch.cat([torch.ones(1, padding_needed), fine_padding], dim=1)
        else:
            padding_mask = None

        return super().generate_tokens(sequence, padding_mask, num_tokens)
