from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
try:
    import pytorch_lightning as pl
except ImportError:
    class pl:
        class LightningModule:
            pass

        class Callback:
            pass
from itertools import chain
import random
import torchaudio

class ResNet1d(nn.Module):
    def __init__(
        self,
        n_channels,
        kernel_size: int = 7,
        padding: str = 'valid',
        dilation: int = 1
    ) -> None:
        super().__init__()
        assert padding in ['valid', 'same']
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self._padding_size = (kernel_size // 2) * dilation
        self.conv0 = nn.Conv1d(
            n_channels,
            n_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation)
        self.conv1 = nn.Conv1d(
            n_channels,
            n_channels,
            kernel_size=1)

    def forward(self, input):
        y = input
        x = self.conv0(input)
        x = F.elu(x)
        x = self.conv1(x)
        if self.padding == 'valid':
            y = y[:, :, self._padding_size:-self._padding_size]
        x += y
        x = F.elu(x)
        return x
    
class EncoderBlock(nn.Module):
    def __init__(
        self,
        n_channels: int,
        padding: str,
        stride: int
    ) -> None:
        super().__init__()
        assert padding in ['valid', 'same']
        self.layers = nn.Sequential(
            ResNet1d(n_channels // 2, padding=padding, dilation=1),
            ResNet1d(n_channels // 2, padding=padding, dilation=3),
            ResNet1d(n_channels // 2, padding=padding, dilation=9),
            nn.Conv1d(
                n_channels // 2, n_channels,
                kernel_size=2 * stride,
                padding=(2 * stride) // 2 if padding == 'same' else 0,
                stride=stride),
            nn.ELU(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)

class Encoder(nn.Module):
    def __init__(self, n_channels: int, padding):
        super().__init__()
        assert padding in ['valid', 'same']
        self.layers = nn.Sequential(
            nn.Conv1d(1, n_channels, kernel_size=7, padding=padding),
            nn.ELU(),
            EncoderBlock(2 * n_channels, padding=padding, stride=2),
            EncoderBlock(4 * n_channels, padding=padding, stride=4),
            EncoderBlock(8 * n_channels, padding=padding, stride=5),
            EncoderBlock(16 * n_channels, padding=padding, stride=8),
            nn.Conv1d(16 * n_channels, 16 * n_channels, kernel_size=3, padding=padding),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)
    
class DecoderBlock(nn.Module):
    def __init__(
        self,
        n_channels: int,
        padding: str,
        stride: int
    ) -> None:
        super().__init__()
        assert padding in ['valid', 'same']
        self.layers = nn.Sequential(
            nn.ConvTranspose1d(
                n_channels, n_channels // 2,
                kernel_size=2 * stride,
                padding=(2 * stride) // 2 if padding == 'same' else 0,
                stride=stride),
            nn.ELU(),
            ResNet1d(n_channels // 2, padding=padding, dilation=1),
            ResNet1d(n_channels // 2, padding=padding, dilation=3),
            ResNet1d(n_channels // 2, padding=padding, dilation=9),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)


class Decoder(nn.Module):
    def __init__(self, n_channels: int, padding):
        super().__init__()
        assert padding in ['valid', 'same']
        self.layers = nn.Sequential(
            nn.Conv1d(16 * n_channels, 16 * n_channels, kernel_size=7, padding=padding),
            nn.ELU(),
            DecoderBlock(16 * n_channels, padding=padding, stride=8),
            DecoderBlock(8 * n_channels, padding=padding, stride=5),
            DecoderBlock(4 * n_channels, padding=padding, stride=4),
            DecoderBlock(2 * n_channels, padding=padding, stride=2),
            nn.Conv1d(n_channels, 1, kernel_size=7, padding=padding),
            nn.Tanh(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input)
    

class ResidualVectorQuantizer(nn.Module):
    weight: torch.Tensor
    running_mean: torch.Tensor
    code_count: torch.Tensor

    def __init__(
        self,
        num_quantizers: int,
        num_embeddings: int,
        embedding_dim: int,
        decay: float = 0.99,
        code_replace_threshold: float = 0.0001,
        eps: float = 1e-10,
    ) -> None:
        super().__init__()
        self.num_quantizers = num_quantizers
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.register_buffer("weight", torch.empty(num_quantizers, num_embeddings, embedding_dim))
        self.register_buffer("running_mean", torch.empty(num_quantizers, num_embeddings, embedding_dim))
        self.register_buffer("code_count", torch.empty(num_quantizers, num_embeddings))
        self.decay = decay
        self.eps = eps
        self.code_replace_threshold = code_replace_threshold
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.uniform_(self.weight)
        self.running_mean[:] = self.weight
        init.ones_(self.code_count)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        # input: [..., chennel]
        if self.training:
            # Enabling bitrate scalability with quantizer dropout
            n = random.randrange(1, self.num_quantizers)
        else:
            n = self.num_quantizers
        codes = []
        r = input.type_as(self.running_mean).detach()
        with torch.no_grad():
            for i in range(n):
                w = self.weight[i]
                # r: [..., num_embeddings]
                dist = torch.cdist(r, w)
                k = torch.argmin(dist, axis=-1)
                codes.append(k)
                self._update_averages(i, r, k)
                r = r - F.embedding(k, w)
        quantized = input - r
        commitment_loss = torch.mean(torch.square(input - quantized.detach()))
        self.weight.data[:] = self.running_mean / torch.unsqueeze(self.eps + self.code_count, axis=-1)
        return quantized, torch.stack(codes, input.ndim - 1), commitment_loss

    def dequantize(self, input: torch.Tensor, n: Optional[int] = None) -> torch.Tensor:
        # input: [batch_size, length, num_quantizers]
        if n is None:
            n = input.shape[-1]
        assert 0 < n <= self.num_quantizers
        res = 0
        with torch.no_grad():
            for i in range(n):
                k = input[:, :, i]
                w = self.weight[i]
                res += F.embedding(k, w)
        return res

    def _update_averages(self, i: int, r: torch.Tensor, k: torch.Tensor) -> None:
        # https://arxiv.org/pdf/1906.00446.pdf
        # Generating Diverse High-Fidelity Images with VQ-VAE-2
        # 2.1 Vector Quantized Variational AutoEncode

        # k: [...]
        one_hot_k = F.one_hot(torch.flatten(k), self.num_embeddings).type_as(self.code_count)
        code_count_update = torch.mean(one_hot_k, axis=0)
        self.code_count[i].lerp_(code_count_update, 1 - self.decay)

        # r: [..., embedding_dim]
        r = r.reshape(-1, self.embedding_dim)
        running_mean_update = (one_hot_k.T @ r) / r.shape[0]
        running_mean_update = running_mean_update.to(self.running_mean[i].dtype)

        self.running_mean[i].lerp_(running_mean_update, 1 - self.decay)

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def replace_vectors(self) -> int:
        # https://arxiv.org/pdf/2107.03312.pdf
        # C. Residual Vector Quantizer:

        # The original paper replaces with an input frame randomly
        # sampled within the current batch.
        # Here we replace with random average of running mean instead.
        num_replaced = torch.sum(self.code_count < self.code_replace_threshold).item()
        if num_replaced > 0:
            for i in range(self.num_quantizers):
                mask = self.code_count[i] < self.code_replace_threshold
                # mask: [num_quantizers, num_embeddings]
                w = torch.rand_like(self.code_count[i])
                w /= torch.sum(w)
                self.running_mean[i, mask] = w.type_as(self.running_mean) @ self.running_mean[i]
                self.code_count[i, mask] = w.type_as(self.code_count) @ self.code_count[i]

        return num_replaced

    @torch.no_grad()
    def calc_entropy(self) -> float:
        p = self.code_count / (self.eps + torch.sum(self.code_count, axis=-1, keepdim=True))
        return -torch.sum(torch.log(p) * p).item() / self.num_quantizers
    

class SoundStreamModel(nn.Module):
    def __init__(
        self,
        n_channels: int = 32,
        num_quantizers: int = 8,
        num_embeddings: int = 1024,
        padding: str = "same"
    ):
        super().__init__()
        self.encoder = Encoder(n_channels, padding)
        self.decoder = Decoder(n_channels, padding)
        self.quantizer = ResidualVectorQuantizer(
            num_quantizers, num_embeddings, n_channels * 16)

    def forward(self, x):
        return self.encode(x)


    def encode(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 2
        x = torch.unsqueeze(input, 1)
        x = self.encoder(x)
        x = torch.transpose(x, -1, -2)
        _, codes, _ = self.quantizer(x)
        return codes

    def decode(self, input: torch.Tensor) -> torch.Tensor:
        # input: [batch_size, length, num_quantizers]
        x = self.quantizer.dequantize(input)
        x = torch.transpose(x, -1, -2)
        x = self.decoder(x)
        x = torch.squeeze(x, 1)
        return x


def soundstream_16khz(pretrained=False, **kwargs):
    """SoundStream encoder decoder

    pretrained (bool): kwargs, load pretrained weights into the model
    """
    # Call the model, load pretrained weights
    model = SoundStreamModel()
    state_dict = torch.hub.load_state_dict_from_url("https://github.com/kaiidams/soundstream-pytorch/releases/download/v1.0/soundstream_16khz-20230425.ckpt", map_location='cpu')
    model.load_state_dict(state_dict['state_dict'], strict=False)
    model.eval()
    return model

def encode_audio(audio_wave, sample_rate, model, start = 0, duration = 3):
    
	x, sr = torchaudio.functional.resample(audio_wave, sample_rate, 16000), 16000

	x = x[:, start*16000:(start + duration)*16000]
    
	with torch.no_grad():
		y = model.encode(x)
		
	return y
    
def decode_audio(embedding, model):
	with torch.no_grad():
		audio = model.decode(embedding)
	return audio

def prepare_acoustic_tokens(y, Q = 8, N = 1024):
    
	y = y.squeeze()
	y = y.reshape(y.shape[0]*y.shape[1])
	size = y.size()[0]
	offsets = torch.tensor([(i % Q) * N for i in range(size)])
	full_token_list = y + offsets

	return full_token_list, offsets

    
def divide_tokens(full_token_list, Q = 8, Q_prime = 3):
    size = full_token_list.numel()
    num_rows = size // Q
    full_token_matrix = full_token_list.reshape(num_rows, Q)
    coarse_token_matrix = full_token_matrix[:, :Q_prime]
    fine_token_matrix = full_token_matrix[:, Q_prime:]
    coarse_token_list = coarse_token_matrix.reshape(-1)
    fine_token_list = fine_token_matrix.reshape(-1)
    return coarse_token_list, fine_token_list

def audio_to_tokens(audio_wave, sample_rate, model, start = 0, duration = 3, Q_prime = 3):

    #y = encode_audio(audio_wave, sample_rate, model, start, duration)
    with torch.no_grad():
        y = model.encode(audio_wave)
    
    full_token_list, _ = prepare_acoustic_tokens(y)
    
    return divide_tokens(full_token_list, 8, Q_prime)

def tokens_to_audio(coarse_tokens, fine_tokens, model, removeOffsets = True, Q = 8, Q_prime = 3, N = 1024):
    
    coarse_shaped, fine_shaped = coarse_tokens.reshape((-1,Q_prime)),fine_tokens.reshape((-1,Q - Q_prime))
    embedding = torch.hstack((coarse_shaped, fine_shaped))

    if removeOffsets:
        size = coarse_tokens.shape[0] + fine_tokens.shape[0]
        offsets = torch.tensor([(i % Q) * N for i in range(size)]).reshape((-1,Q))
        # print(embedding.shape, offsets.shape)
        embedding = (embedding - offsets).reshape((1,-1,Q))
    
    return decode_audio(embedding, model)
    
    
    
