"""BTC: the Bi-directional Transformer for Chord Recognition (ISMIR 2019).

Inference-only port of Jonggwon Park's reference implementation
(https://github.com/jayg996/BTC-ISMIR19, MIT licensed, see LICENSE-BTC), kept
faithful enough to load the published checkpoints unchanged. Everything the
checkpoint has a weight for is reproduced exactly, quirks included:

* the layer norm divides by an *unbiased* standard deviation (`torch.std`),
  not the biased one `nn.LayerNorm` uses, so it can't be swapped for the
  built-in module;
* the position-wise feed-forward applies its ReLU after *every* convolution,
  the last one included, so its output is non-negative;
* attention is masked in both directions — one block sees only the past, its
  twin only the future — which is what makes the model "bi-directional".

Dropout is left out (this only ever runs under `eval()`), and so is the unused
LSTM the reference output layer carries: see `load_state_dict` below.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def timing_signal(length: int, channels: int) -> torch.Tensor:
    """Sinusoidal position encoding, [1, length, channels].

    The transformer's original formulation, as ported into BTC from
    tensor2tensor: `channels // 2` sine components followed by the same number
    of cosines, over timescales spanning 1 … 1e4.
    """
    position = np.arange(length)
    num_timescales = channels // 2
    log_increment = math.log(1.0e4) / (num_timescales - 1)
    inv_timescales = np.exp(
        np.arange(num_timescales, dtype=np.float64) * -log_increment
    )
    scaled = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)
    signal = np.concatenate([np.sin(scaled), np.cos(scaled)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]], "constant")
    return torch.from_numpy(signal.reshape([1, length, channels])).float()


def causal_mask(length: int) -> torch.Tensor:
    """Additive attention mask, [1, 1, length, length], hiding future frames.

    Transposing the last two dimensions turns it into the mask that hides the
    past instead — which is exactly how the backward block is built.
    """
    mask = np.triu(np.full([length, length], -np.inf), 1)
    return torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(1)


class LayerNorm(nn.Module):
    """Layer norm over the last dimension, with an unbiased standard deviation.

    `nn.LayerNorm` normalizes by the biased (population) standard deviation.
    BTC was trained against this variant, so the two are not interchangeable:
    swapping it in shifts every activation slightly and costs accuracy.
    """

    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class MultiHeadAttention(nn.Module):
    """Scaled dot-product attention over `num_heads` heads, biased by a mask."""

    def __init__(
        self,
        depth: int,
        key_depth: int,
        value_depth: int,
        num_heads: int,
        bias_mask: torch.Tensor,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.query_scale = (key_depth // num_heads) ** -0.5
        # A buffer, not a plain attribute: it has to follow the model onto the
        # GPU, and it must stay out of the state dict (the checkpoint has no
        # entry for it) — hence persistent=False.
        self.register_buffer("bias_mask", bias_mask, persistent=False)
        self.query_linear = nn.Linear(depth, key_depth, bias=False)
        self.key_linear = nn.Linear(depth, key_depth, bias=False)
        self.value_linear = nn.Linear(depth, value_depth, bias=False)
        self.output_linear = nn.Linear(value_depth, depth, bias=False)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, depth = x.shape
        return x.view(batch, length, self.num_heads, depth // self.num_heads).permute(
            0, 2, 1, 3
        )

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        batch, heads, length, depth = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(batch, length, depth * heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        queries = self._split(self.query_linear(x)) * self.query_scale
        keys = self._split(self.key_linear(x))
        values = self._split(self.value_linear(x))
        logits = torch.matmul(queries, keys.permute(0, 1, 3, 2))
        length = logits.shape[-1]
        logits = logits + self.bias_mask[:, :, :length, :length].type_as(logits)
        weights = F.softmax(logits, dim=-1)
        return self.output_linear(self._merge(torch.matmul(weights, values)))


class Conv(nn.Module):
    """1-D convolution over a [batch, length, channels] sequence, centre-padded."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.pad = nn.ConstantPad1d((kernel_size // 2, (kernel_size - 1) // 2), 0)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pad(x.permute(0, 2, 1))).permute(0, 2, 1)


class PositionwiseFeedForward(nn.Module):
    """BTC's convolutional feed-forward: conv → ReLU → conv → ReLU.

    The trailing ReLU is not a typo — the reference loops over its layers and
    activates after each one, so the block's output is non-negative. Removing
    it changes what the residual stream carries.
    """

    def __init__(self, hidden_size: int, filter_size: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [Conv(hidden_size, filter_size), Conv(filter_size, hidden_size)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = F.relu(layer(x))
        return x


class SelfAttentionBlock(nn.Module):
    """Pre-norm attention + feed-forward, each wrapped in a residual."""

    def __init__(
        self,
        hidden_size: int,
        key_depth: int,
        value_depth: int,
        filter_size: int,
        num_heads: int,
        bias_mask: torch.Tensor,
    ):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(
            hidden_size, key_depth, value_depth, num_heads, bias_mask
        )
        self.positionwise_convolution = PositionwiseFeedForward(
            hidden_size, filter_size
        )
        self.layer_norm_mha = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.multi_head_attention(self.layer_norm_mha(x))
        return x + self.positionwise_convolution(self.layer_norm_ffn(x))


class BiDirectionalSelfAttention(nn.Module):
    """One BTC layer: a past-only and a future-only block, then a projection."""

    def __init__(
        self,
        hidden_size: int,
        key_depth: int,
        value_depth: int,
        filter_size: int,
        num_heads: int,
        max_length: int,
    ):
        super().__init__()
        forward_mask = causal_mask(max_length)
        params = (hidden_size, key_depth, value_depth, filter_size, num_heads)
        self.attn_block = SelfAttentionBlock(*params, forward_mask)
        self.backward_attn_block = SelfAttentionBlock(
            *params, torch.transpose(forward_mask, 2, 3)
        )
        self.linear = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        both = torch.cat((self.attn_block(x), self.backward_attn_block(x)), dim=2)
        return self.linear(both)


class BiDirectionalSelfAttentionLayers(nn.Module):
    """The encoder stack: project the features in, add timing, run the layers."""

    def __init__(
        self,
        feature_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        key_depth: int,
        value_depth: int,
        filter_size: int,
        max_length: int,
    ):
        super().__init__()
        self.register_buffer(
            "timing_signal", timing_signal(max_length, hidden_size), persistent=False
        )
        self.embedding_proj = nn.Linear(feature_size, hidden_size, bias=False)
        self.self_attn_layers = nn.ModuleList(
            BiDirectionalSelfAttention(
                hidden_size,
                key_depth,
                value_depth,
                filter_size,
                num_heads,
                max_length,
            )
            for _ in range(num_layers)
        )
        self.layer_norm = LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding_proj(x)
        x = x + self.timing_signal[:, : x.shape[1], :].type_as(x)
        for layer in self.self_attn_layers:
            x = layer(x)
        return self.layer_norm(x)


class OutputLayer(nn.Module):
    """Linear projection from the encoder's hidden size to the chord vocabulary."""

    def __init__(self, hidden_size: int, output_size: int):
        super().__init__()
        self.output_projection = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_projection(x)


# Architecture of the published checkpoints (run_config.yaml in the reference
# repo). `timestep` is both the attention mask's size and the number of frames
# the model is fed at a time, so it is a hard constraint on the caller, not a
# tunable: muscriptor.utils.chords sizes its feature blocks around it.
FEATURE_SIZE = 144
TIMESTEP = 108
HIDDEN_SIZE = 128
NUM_LAYERS = 8
NUM_HEADS = 4
FILTER_SIZE = 128
# The "large vocabulary" head: 12 roots x 14 qualities, plus X (unknown) and N.
NUM_CHORDS_LARGE_VOCA = 170


class BTCModel(nn.Module):
    """The published BTC network, minus everything only training needed."""

    def __init__(self, num_chords: int = NUM_CHORDS_LARGE_VOCA):
        super().__init__()
        self.self_attn_layers = BiDirectionalSelfAttentionLayers(
            FEATURE_SIZE,
            HIDDEN_SIZE,
            NUM_LAYERS,
            NUM_HEADS,
            HIDDEN_SIZE,
            HIDDEN_SIZE,
            FILTER_SIZE,
            TIMESTEP,
        )
        self.output_layer = OutputLayer(HIDDEN_SIZE, num_chords)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Chord logits for `features` [batch, TIMESTEP, FEATURE_SIZE]."""
        return self.output_layer(self.self_attn_layers(features))

    def load_published_state_dict(self, state_dict: dict) -> None:
        """Load a reference checkpoint's `model` dict into this network.

        The reference output layer allocates a bidirectional LSTM that its
        forward pass never calls, so the checkpoint carries weights this port
        has no module for. They are dropped here rather than loaded with
        `strict=False`, which would also swallow a genuinely mismatched
        checkpoint.
        """
        wanted = {k: v for k, v in state_dict.items() if "output_layer.lstm." not in k}
        self.load_state_dict(wanted)
