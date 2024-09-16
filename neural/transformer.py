import torch
from torch import nn, Tensor
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor):
        x = x + self.pe[:, : x.size(1)]
        return x


class PE_Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        max_len: int,
    ):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            nhead - Number of attention heads.
            num_encoder_layers - Number of transformer layers in the encoder.
            num_decoder_layers - Number of transformer layers in the decoder.
            dim_feedforward - Hidden dimensionality of the feedforward network in the transformer.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create positional encodings
        self.pe = PositionalEncoding(d_model, max_len)

        # Create transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
        )

    def forward(self, src: Tensor, tgt: Tensor):
        src = self.pe(src)
        tgt = self.pe(tgt)
        out = self.transformer(src, tgt)
        return out


def gen_pe(max_length, d_model, n):

    # generate an empty matrix for the positional encodings (pe)
    pe = torch.zeros(max_length * d_model).reshape(max_length, d_model)

    # for each position
    for k in torch.arange(max_length):

        # for each dimension
        for i in torch.arange(d_model // 2):

            # calculate the internal value for sin and cos
            theta = k / (n ** ((2 * i) / d_model))

            # even dims: sin
            pe[k, 2 * i] = math.sin(theta)

            # odd dims: cos
            pe[k, 2 * i + 1] = math.cos(theta)

    return pe
