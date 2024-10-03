import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import pytorch_lightning as pl
from typing import List
import tqdm

class RelativePosition(nn.Module):

    def __init__(self, d_a: int, k: int, myDevice: torch.device):
        super().__init__()
        self.d_a = d_a
        self.k = k
        self.myDevice = myDevice
        self.position_embeddings = nn.Parameter(torch.empty((2 * k + 1, d_a), device=self.myDevice))
        nn.init.xavier_uniform_(self.position_embeddings)

    def forward(self, length_query: int, length_key: int) -> torch.Tensor:
        indices_query = torch.arange(length_query, device=self.myDevice)
        indices_key = torch.arange(length_key, device=self.myDevice)
        distance_matrix = indices_key.unsqueeze(0) - indices_query.unsqueeze(1)
        distance_matrix_clipped = torch.clamp(distance_matrix, -self.k, self.k)
        final_matrix = distance_matrix_clipped + self.k
        embeddings = self.position_embeddings[final_matrix.to(torch.long)]
        return embeddings


class AttentionHead(nn.Module):

    def __init__(self, hidden_size, d_model, myDevice: torch.device):
        super().__init__()
        self.d_model = d_model
        self.myDevice = myDevice
        self.query_weights: nn.Linear = nn.Linear(hidden_size, self.d_model).to(myDevice)
        self.key_weights: nn.Linear = nn.Linear(hidden_size, self.d_model).to(myDevice)
        self.value_weights: nn.Linear = nn.Linear(hidden_size, self.d_model).to(myDevice)

    def forward(self, queryIn: torch.Tensor, keyIn: torch.Tensor, valueIn: torch.Tensor, maskIn: torch.Tensor = None, k_bias_matrix: torch.Tensor = None, v_bias_matrix: torch.Tensor = None) -> torch.Tensor:
        query: torch.Tensor = self.query_weights(queryIn) 
        key: torch.Tensor = self.key_weights(keyIn) 
        value: torch.Tensor = self.value_weights(valueIn)

        attn_1: torch.Tensor = torch.matmul(query, key.transpose(1, 2))

        if k_bias_matrix is None:
            k_bias_matrix = torch.zeros_like(attn_1, device=self.myDevice)

        attn_2: torch.Tensor = torch.matmul(query.transpose(0, 1), k_bias_matrix.transpose(1, 2)).transpose(0, 1)
        att_scores: torch.Tensor = (attn_1 + attn_2) / (self.d_model ** 0.5)

        if maskIn is not None:
            mask = maskIn.to(torch.int).to(self.myDevice)
            att_scores_filled: torch.Tensor = att_scores.masked_fill(mask == 0, -1e9)
        else:
            att_scores_filled: torch.Tensor = att_scores

        att_weights: torch.Tensor = F.softmax(att_scores_filled, dim=-1)
        values_1: torch.Tensor = torch.matmul(att_weights, value)

        if v_bias_matrix is None:
            v_bias_matrix = torch.zeros_like(values_1, device=self.myDevice)

        values_2: torch.Tensor = torch.matmul(att_weights.transpose(0, 1), v_bias_matrix).transpose(0, 1)
        n_value = values_1 + values_2
        return n_value


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size=1024, num_heads=16, k=64, seq_len=1500, myDevice: torch.device = torch.device("cpu")):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size: int = hidden_size
        self.num_heads: int = num_heads
        self.head_dim: int = hidden_size // num_heads
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.myDevice = myDevice

        self.relative_position_k = RelativePosition(self.head_dim, k, myDevice)
        self.relative_position_v = RelativePosition(self.head_dim, k, myDevice)

        self.attention_heads = nn.ModuleList([AttentionHead(self.hidden_size, self.head_dim, myDevice) for _ in range(self.num_heads)])
        self.fc = nn.Linear(hidden_size, hidden_size).to(myDevice)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        k_bias_matrix = self.relative_position_k(self.seq_len, self.seq_len)
        v_bias_matrix = self.relative_position_v(self.seq_len, self.seq_len)

        attention_outputs: List[torch.Tensor] = [attention_head(query, key, value, mask, k_bias_matrix, v_bias_matrix)
                                                 for attention_head in self.attention_heads]

        hidden_state: torch.Tensor = torch.cat(attention_outputs, dim=-1)
        hidden_state_final: torch.Tensor = self.fc(hidden_state)

        return hidden_state_final


class DecoderLayer(nn.Module):
    def __init__(self, d_model=1024, num_heads=16, dim_feedforward=4096, dropout=0.1, k=64, seq_len=1500, myDevice: torch.device = torch.device("cpu")):
        super(DecoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, k, seq_len, myDevice)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward).to(myDevice),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model).to(myDevice)
        )
        self.layer_norm1 = nn.LayerNorm(d_model).to(myDevice)
        self.layer_norm2 = nn.LayerNorm(d_model).to(myDevice)
        self.dropout = nn.Dropout(dropout)
        self.myDevice = myDevice

    def forward(self, x, mask: torch.Tensor = None):
        attn_output = self.attention(x, x, x, mask)
        x_attended = self.layer_norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x_attended)
        x_final = self.layer_norm2(x_attended + self.dropout(ff_output))
        return x_final
    
class Decoder(pl.LightningModule):
    def __init__(self, d_model=1024, num_layers=12, num_heads=16, dim_feedforward=4096, dropout=0.1, k=64, seq_len=1500, vocab_size=1500, learning_rate=10e-4, myDevice=torch.device("cpu")):
        super().__init__()
        self.seq_len = seq_len
        self.myDevice = myDevice
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model).to(myDevice)
        
        # Stacking decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, dim_feedforward, dropout, k, seq_len, myDevice)
            for _ in range(num_layers)
        ])
        
        # Layer normalization and final linear layer
        self.norm = nn.LayerNorm(d_model).to(myDevice)
        self.linear = nn.Linear(d_model, vocab_size).to(myDevice)
        
        # Loss function and optimizer
        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss().to(myDevice)
        self.optimizer = self.configure_optimizers()    
        
        # Disable automatic optimization in PyTorch Lightning
        self.automatic_optimization = False
        
        # Causal mask
        self.causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=myDevice)).unsqueeze(0).detach()
        self.causal_mask.requires_grad = False

    def forward(self, x, mask: torch.Tensor = None):
        # Forward pass through embedding and layers
        x = self.embedding(x).to(self.myDevice)
        for layer in self.layers:
            x = layer(x, mask)
        output = self.linear(self.norm(x))
        return output
    
    def common_step(self, batch):
        input_ids, target_ids = batch
        input_ids = input_ids.to(self.myDevice)
        target_ids = target_ids.to(self.myDevice)
        
        # Causal mask
        output = self(input_ids, self.causal_mask)
        
        # Reshaping for cross-entropy loss calculation
        batch_size, input_seq_len = input_ids.shape
        target_seq_len = target_ids.shape[1]
        output_target = output[:, input_seq_len - target_seq_len:, :]
        
        # Reshape for CrossEntropyLoss
        output_target_reshaped = output_target.reshape(-1, output_target.size(-1))
        target_ids_reshaped = target_ids.reshape(-1)
        
        # Compute loss
        loss = self.loss_fn(output_target_reshaped, target_ids_reshaped)
        return loss, output_target_reshaped, target_ids_reshaped
    
    def training_step(self, batch, batch_idx):
        self.optimizer.zero_grad()
        loss, _, _ = self.common_step(batch)
        self.log('train_loss', loss)
        
        with torch.autograd.set_detect_anomaly(True):
            self.manual_backward(loss)
        self.optimizer.step()
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss, output_target, target_ids = self.common_step(batch)
        labels_hat = torch.argmax(output_target, dim=1)
        val_acc = torch.sum(target_ids == labels_hat).item() / len(target_ids)
        self.log_dict({'val_loss': loss, 'val_acc': val_acc})
        return {'val_loss': loss, 'val_acc': val_acc}

    def on_train_epoch_end(self):
        torch.save(self.state_dict(), f"checkpoints/{self.__class__.__name__}_epoch={self.current_epoch:02d}.ckpt")

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        return optimizer

    def generate_tokens(self, input_ids, padding_mask=None, num_tokens=10, cut_value = 0):
        input_ids = input_ids.to(self.myDevice)
        generated_sequence = input_ids.unsqueeze(0).to(self.myDevice)
        total_sequence = generated_sequence
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).to(self.myDevice)
        
        for _ in range(num_tokens):
            with torch.no_grad():
                if padding_mask is not None:
                    combined_mask = self.causal_mask * padding_mask
                    padding_mask = torch.cat([padding_mask[:, :, 1:], torch.tensor([[[1]]], device=self.myDevice)], dim=2)
                else:
                    combined_mask = self.causal_mask
                
                output = self(generated_sequence, combined_mask)
                next_token_logits = output[:, -1, :]
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.argmax(probs, dim=-1)

                first_free_spot = (padding_mask[0, 0, cut_value:] == 0).nonzero(as_tuple=True)
                if len(first_free_spot[0]) > 0:
                    position = cut_value + first_free_spot[0][0]
                    total_sequence[0, position] = next_token
                    padding_mask[0, 0, position] = 1
                else:
                    # If no free spot, append to the end
                    total_sequence = torch.cat((total_sequence, next_token.unsqueeze(0)), dim=1)

                generated_sequence = torch.cat((generated_sequence[:, 1:], next_token.unsqueeze(0)), dim=1)

        return total_sequence

    def pad_sequence(self, sequence, length):
        to_pad = length - sequence.shape[0]
        if to_pad > 0:
            padded_sequence = F.pad(sequence, (0, to_pad)).to(self.myDevice)
            padding_mask = torch.ones(1, length, device=self.myDevice)
            padding_mask[0, sequence.shape[0]:] = 0
            return padded_sequence, padding_mask
        elif to_pad == 0:
            return sequence.to(self.myDevice), None
        else:
            raise ValueError(f"Invalid token length, shorten the input to match this size {length}")


class SemanticTransformer(Decoder):
    def __init__(self, d_model=1024, num_layers=12, num_heads=16, dim_feedforward=4096, dropout=0.1, k=64, audioDuration=30, vocab_size=500, learning_rate=10e-4, myDevice=torch.device("cpu")):
        self.seq_len = int(50 * audioDuration - 2)
        super().__init__(d_model, num_layers, num_heads, dim_feedforward, dropout, k, self.seq_len, vocab_size, learning_rate, myDevice)
        self.myDevice = myDevice
    
    def get_seq_len(self):
        return self.seq_len

    def generate_tokens(self, semantic_tokens, num_tokens=None):
        semantic_tokens = semantic_tokens.to(self.myDevice)
        if num_tokens == None:
            num_tokens = semantic_tokens.shape[0] - self.seq_len
        padded_semantic_tokens, padding_mask = self.pad_sequence(semantic_tokens, self.seq_len)
        return super().generate_tokens(padded_semantic_tokens, padding_mask, num_tokens)

    # Override of Decoder.common_step
    def common_step(self, batch):
        input_ids, target_ids = batch
        input_ids = input_ids.to(self.myDevice)
        target_ids = target_ids.to(self.myDevice)

        combined_mask = self.causal_mask * (input_ids != 0).float().unsqueeze(1).expand(-1, self.seq_len, self.seq_len)
        output = self(input_ids, combined_mask)
        
        batch_size, input_seq_len = input_ids.shape
        target_seq_len = target_ids.shape[1]
        output_target = output[:, input_seq_len - target_seq_len:, :]

        # Reshape for CrossEntropyLoss
        output_target = output_target.reshape(-1, output_target.size(-1))
        target_ids = target_ids.reshape(-1)

        loss = self.loss_fn(output_target, target_ids)
        return loss, output_target, target_ids

        
class CoarseTransformer(Decoder):
    def __init__(self, d_model=1024, num_layers=12, num_heads=16, dim_feedforward=4096, dropout=0.1, k=64, audioDuration=10, Q_prime=3, vocab_size=3072, learning_rate=10e-4, myDevice=torch.device("cpu")):
        self.semantic_size = int(50 * audioDuration - 1)
        self.coarse_size = int((50 * audioDuration + 1) * Q_prime) - 1
        self.seq_len = self.semantic_size + self.coarse_size
        self.myDevice = myDevice
        self.Q_prime = Q_prime
        
        super().__init__(d_model, num_layers, num_heads, dim_feedforward, dropout, k, self.seq_len, vocab_size, learning_rate, myDevice)

    def get_semantic_size(self):
        return self.semantic_size
    def get_coarse_size(self):
        return self.coarse_size
    
    def generate_tokens(self, semantic_tokens, coarse_tokens, num_tokens=None):
        # Ensure tokens are moved to the correct myDevice
        semantic_tokens = semantic_tokens.to(self.myDevice)
        coarse_tokens = coarse_tokens.to(self.myDevice)
        
        if semantic_tokens.shape[0] > self.semantic_size:
            semantic_tokens = semantic_tokens[:self.semantic_size]

        if num_tokens == None:
            num_tokens = coarse_tokens.shape[0] - self.coarse_size
        padded_semantic_tokens, semantic_padding = self.pad_sequence(semantic_tokens, self.semantic_size)
        padded_coarse_tokens, coarse_padding = self.pad_sequence(coarse_tokens, self.coarse_size)
        sequence = torch.cat((padded_semantic_tokens, padded_coarse_tokens), dim=0).to(self.myDevice)

        if semantic_padding is not None and coarse_padding is not None:
            padding_mask = torch.cat((semantic_padding, coarse_padding), dim=1).to(self.myDevice)
        elif semantic_padding is not None:
            padding_needed = self.seq_len - semantic_padding.shape[1]
            padding_mask = torch.cat([semantic_padding, torch.ones(1, padding_needed, device=self.myDevice)], dim=1)
        elif coarse_padding is not None:
            padding_needed = self.seq_len - coarse_padding.shape[1]
            padding_mask = torch.cat([torch.ones(1, padding_needed, device=self.myDevice), coarse_padding], dim=1)
        else:
            padding_mask = None

        total_token_sequence = super().generate_tokens(sequence, padding_mask, num_tokens, self.semantic_size)
        coarse_sequence = total_token_sequence[:, self.semantic_size:]
        while coarse_sequence.shape[1] % self.Q_prime != 0:
            coarse_sequence = coarse_sequence[:, :-1]
        return coarse_sequence
        

    # Override of Decoder.common_step
    def common_step(self, batch):
        input_ids, target_ids = batch
        input_ids = input_ids.to(self.myDevice)
        target_ids = target_ids.to(self.myDevice)

        combined_mask = self.causal_mask * (input_ids != 0).float().unsqueeze(1).expand(-1, self.seq_len, self.seq_len).to(self.myDevice)

        output = self(input_ids, combined_mask)  # output shape: (batch_size, seq_len, vocab_size)

        batch_size, input_seq_len = input_ids.shape
        target_seq_len = target_ids.shape[1]

        output_target = output[:, input_seq_len - target_seq_len:, :]  # I take only the tokens I need to check (for the coarse generation I take only the coarse generated)

        # Reshape for CrossEntropyLoss
        output_target = output_target.reshape(-1, output_target.size(-1))  # (batch_size * target_seq_len, vocab_size)
        target_ids = target_ids.reshape(-1)

        # Compute loss
        loss = self.loss_fn(output_target, target_ids)
        return loss, output_target, target_ids


class FineTransformer(Decoder):
    def __init__(self, d_model=1024, num_layers=12, num_heads=16, dim_feedforward=4096, dropout=0.1, k=64, audioDuration=3, Q_prime=3, Q=8, vocab_size=8192, learning_rate=10e-4, myDevice=torch.device("cpu")):
        self.coarse_size = int((50 * audioDuration + 1) * Q_prime)
        self.fine_size = int((50 * audioDuration + 1) * (Q - Q_prime) - 1)
        self.seq_len = self.coarse_size + self.fine_size
        self.myDevice = myDevice
        self.Q = Q
        self.Q_prime = Q_prime
        super().__init__(d_model, num_layers, num_heads, dim_feedforward, dropout, k, self.seq_len, vocab_size, learning_rate, myDevice)

    def get_coarse_size(self):
        return self.coarse_size
    def get_fine_size(self):
        return self.fine_size
    
    def generate_tokens(self, coarse_tokens, fine_tokens, num_tokens=None):
        # Ensure tokens are moved to the correct myDevice
        coarse_tokens = coarse_tokens.to(self.myDevice)
        fine_tokens = fine_tokens.to(self.myDevice)

        if coarse_tokens.shape[0] > self.coarse_size:
            coarse_tokens = coarse_tokens[:self.coarse_size]

        if num_tokens == None:
            num_tokens = fine_tokens.shape[0] - self.fine_size
        padded_coarse_tokens, coarse_padding = self.pad_sequence(coarse_tokens, self.coarse_size)
        padded_fine_tokens, fine_padding = self.pad_sequence(fine_tokens, self.fine_size)
        sequence = torch.cat((padded_coarse_tokens, padded_fine_tokens), dim=0).to(self.myDevice)

        if fine_padding is not None and coarse_padding is not None:
            padding_mask = torch.cat((coarse_padding, fine_padding), dim=1).to(self.myDevice)
        elif coarse_padding is not None:
            padding_needed = self.seq_len - coarse_padding.shape[1]
            padding_mask = torch.cat([coarse_padding, torch.ones(1, padding_needed, device=self.myDevice)], dim=1)
        elif fine_padding is not None:
            padding_needed = self.seq_len - fine_padding.shape[1]
            padding_mask = torch.cat([torch.ones(1, padding_needed, device=self.myDevice), fine_padding], dim=1)
        else:
            padding_mask = None

        total_token_sequence = super().generate_tokens(sequence, padding_mask, num_tokens, self.coarse_size)
        fine_sequence = total_token_sequence[:, self.coarse_size:]
        while fine_sequence.shape[1] % (self.Q - self.Q_prime) != 0:
            fine_sequence = fine_sequence[:, :-1]
        return fine_sequence


def generate_new_sequence(semantic_tokens, coarse_tokens, fine_tokens, semantic_model, coarse_model, fine_model, audioDuration, Q = 8, Q_prime = 3):

    semanticlength = int(50 * audioDuration)
    coarselength = int((50 * audioDuration + 1) * Q_prime)
    finelength = int((50 * audioDuration + 1) * (Q - Q_prime))
    
    with torch.no_grad():

        print("Generating semantic tokens...")

        semantic_to_generate = semanticlength - semantic_tokens.shape[0]
        max_sem = semantic_model.get_seq_len()
        if semantic_to_generate > 0:
            i = semantic_tokens.shape[0]//max_sem
            current_semantic =  semantic_tokens[i*max_sem:]
            new_semantic = semantic_model.generate_tokens(current_semantic, semantic_to_generate).squeeze(0)
            total_semantic = torch.cat([semantic_tokens[:i*max_sem], new_semantic], dim=0)
        else:
            total_semantic = semantic_tokens

        print("Generating coarse tokens...")

        coarse_to_generate = coarselength - coarse_tokens.shape[0]
        max_coarse = coarse_model.get_coarse_size()
        max_sem = coarse_model.get_semantic_size()
        if coarse_to_generate > 0:
            i = coarse_tokens.shape[0]//max_coarse
            current_coarse = coarse_tokens[i*max_coarse:]
            current_semantic = total_semantic[i*max_sem:(i+1)*max_sem]
            while coarse_to_generate > 0:
                new_coarse = coarse_model.generate_tokens(current_semantic, current_coarse).squeeze(0)
                generated = new_coarse.shape[0] - current_coarse.shape[0] 
                coarse_to_generate = coarse_to_generate - generated
                coarse_tokens = torch.cat([coarse_tokens, new_coarse[current_coarse.shape[0]:]], dim=0)
                current_coarse = new_coarse[current_coarse.shape[0]:]
                i = i + 1
                if i*max_sem < total_semantic.shape[0]:
                    current_semantic = total_semantic[i*max_sem:(i+1)*max_sem]
                else:
                    break

        print("Generating fine tokens...")

        fine_to_generate = finelength - fine_tokens.shape[0]
        max_coarse = fine_model.get_coarse_size()
        max_fine = fine_model.get_fine_size()
        if fine_to_generate > 0:
            i = fine_tokens.shape[0]//max_fine
            current_fine = fine_tokens[i*max_fine:]
            current_coarse = coarse_tokens[i*max_coarse:(i+1)*max_coarse]
            while fine_to_generate > 0:
                new_fine = fine_model.generate_tokens(current_coarse, current_fine).squeeze(0)
                generated = new_fine.shape[0] - current_fine.shape[0] 
                fine_to_generate = fine_to_generate - generated
                fine_tokens = torch.cat([fine_tokens, new_fine[current_fine.shape[0]:]], dim=0)
                current_fine = new_fine[current_fine.shape[0]:]
                i = i + 1
                if i*max_coarse < coarse_tokens.shape[0]:
                    current_coarse = coarse_tokens[i*max_coarse:(i+1)*max_coarse]
                else:
                    break
        
        coarse_tokens = coarse_tokens[:coarselength]
        fine_tokens = fine_tokens[:finelength]
        
        return coarse_tokens, fine_tokens
