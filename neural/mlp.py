import torch
from torch import Tensor, nn
from typing import Union, Sequence
from lightning import LightningModule

import neural.module as neural


class MLPLayer(nn.Module):
    """
    A single layer perceptron, that can hold a bach-norm and activation layers as well.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        nonlin: Union[str, nn.Module],
        batch_norm: bool = False,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        layers = []

        layers.append(nn.Linear(in_dim, out_dim))
        in_dim = out_dim
        if batch_norm and nonlin not in ["none", None]:
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(self._make_activation(nonlin))

        self.mlp_layer = nn.Sequential(*layers)

    def _make_activation(self, act: Union[str, nn.Module]) -> nn.Module:
        if isinstance(act, str):
            return neural.ACTIVATIONS[act](**neural.ACTIVATION_DEFAULT_KWARGS[act])
        return act

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: An input tensor, of shape (N, D) containing N samples with D features.

        Returns:
            An output tensor of shape (N, D_out) where D_out is the output dim.

        """
        # return self.mlp_layer.forward(x.reshape(x.size(0), -1))
        # return self.mlp_layer.forward(x)
        # shape = x.shape
        # print(x.dtype)
        res = self.mlp_layer.forward(x)
        # res = res.reshape(shape)
        return res



class MlpBlock(nn.Module):
    """
    A general-purpose MLP.

    Args:
        in_dim: Input dimension.
        dims: Hidden dimensions, including output dimension.
        nonlins: Non-linearities to apply after each one of the hidden
            dimensions.
            Can be either a sequence of strings which are keys in the ACTIVATIONS
            dict, or instances of nn.Module (e.g. an instance of nn.ReLU()).
            Length should match 'dims'.
    """

    def __init__(
        self,
        in_dim: int,
        dims: Sequence[int],
        nonlins: Sequence[Union[str, nn.Module]],
        batch_norm: bool = False,
    ):
        assert len(nonlins) == len(dims)
        self.in_dim = in_dim
        self.out_dim = dims[-1]
        self.dims = dims
        self.nonlins = nonlins

        super().__init__()

        layers = []
        for i, out_dim in enumerate(self.dims):
            layers.append(MLPLayer(in_dim, out_dim, nonlins[i], batch_norm))
            in_dim = out_dim

        self.sequence = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: An input tensor, of shape (N, D) containing N samples with D features.

        Returns:
            An output tensor of shape (N, D_out) where D_out is the output dim.
        """
        return self.sequence.forward(x)


class RMLP(nn.Module):
    def __init__(
        self,
        block_in_dim: int,
        block_dims: Sequence[int],
        block_nonlins: Sequence[Union[str, nn.Module]],
        n_blocks: int,
        out_dim: int,
        out_nonlin: Union[str, nn.Module],
        in_dim: int = None,  # if in_dim is an int, then a first layer will be made
        batch_norm: bool = False,
    ) -> None:
        super().__init__()

        # Create first layer if in_dim is not None
        self.input = nn.Identity()
        if in_dim is not None:
            self.input = MLPLayer(in_dim, block_in_dim, block_nonlins[0], batch_norm)

        # Create blocks
        layers = []
        for i in range(n_blocks):
            layers.append(MlpBlock(block_in_dim, block_dims, block_nonlins, batch_norm))

        self.blocks = nn.ModuleList(layers)
        # Create output layer
        # self.output = nn.Linear(block_dims[-1], out_dim)
        self.output = MLPLayer(block_dims[-1], out_dim, out_nonlin, False)

    def _make_activation(self, act: Union[str, nn.Module]) -> nn.Module:
        if isinstance(act, str):
            return neural.ACTIVATIONS[act](**neural.ACTIVATION_DEFAULT_KWARGS[act])
        return act

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: An input tensor, of shape (N, D) containing N samples with D features.

        Returns:
            An output tensor of shape (N, D_out) where D_out is the output dim.
        """
        x = self.input(x)
        for block in self.blocks:
            out = block(x)
            x = x + out
        return self.output(x)


class MLPBinaryClassifier(neural.LightningModel):
    def __init__(
        self,
        in_dim: int,
        dims: Sequence[int],
        nonlins: Sequence[Union[str, nn.Module]],
        batch_norm: bool = False,
    ):
        super().__init__()
        self.model: nn.Module = MlpBlock(in_dim, dims, nonlins, batch_norm)
        self.output = MLPLayer(dims[-1], 1, "sigmoid", batch_norm)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(-2)
        x = self.model(x)
        return self.output(x)
        # return self.model(x)[..., [0]]

    def accuracy(self, preds: Tensor, labels: Tensor) -> Tensor:
        class_preds = (preds > 0.5).to(torch.int32)
        return (class_preds == labels).float().mean()
    
class MLP_Equiv(neural.LightningModel):
    def __init__(
        self,
        in_dim: int,
        dims: Sequence[int],
        nonlins: Sequence[Union[str, nn.Module]],
        batch_norm: bool = False,
        epsilon: float = 1e-3,
    ):
        super().__init__()
        self.model: nn.Module = MlpBlock(in_dim, dims, nonlins, batch_norm)
        self.output = MLPLayer(dims[-1], in_dim, 'none', batch_norm)
        self.epsilon = epsilon

    def forward(self, x: Tensor) -> Tensor:
        input = x.flatten(-2)
        out = self.model(input)
        return self.output(out).reshape(x.shape)

    def accuracy(self, preds: Tensor, labels: Tensor) -> Tensor:
        diffs = (preds - labels).abs()  # (batch_size, in_dim, out_channels)
        acc = (diffs.sum(dim=(1))) < self.epsilon  # (batch_size, out_channels)
        acc = acc.sum(dim=1)  # (batch_size)

        return acc.to(torch.float).mean()
    
