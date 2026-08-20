"""Tests for muscriptor/models/btc.py, the ported chord-recognition network.

The published weights are not needed here (they are a 12 MB download): what
these pin down is the shape of the port — the quirks that let it load those
weights and still compute what the reference computed.
"""

import pytest
import torch
from torch import nn

from muscriptor.models.btc import (
    FEATURE_SIZE,
    NUM_CHORDS_LARGE_VOCA,
    TIMESTEP,
    BTCModel,
    LayerNorm,
    causal_mask,
    timing_signal,
)


def test_forward_labels_every_frame():
    model = BTCModel().eval()
    with torch.inference_mode():
        logits = model(torch.zeros(2, TIMESTEP, FEATURE_SIZE))
    assert logits.shape == (2, TIMESTEP, NUM_CHORDS_LARGE_VOCA)


def test_layer_norm_uses_an_unbiased_standard_deviation():
    """The reference normalizes with `torch.std`, which nn.LayerNorm does not.

    They differ by a factor of sqrt(n / (n - 1)), so a "simplification" to the
    built-in module would quietly rescale every activation in the network.
    """
    norm = LayerNorm(4)
    x = torch.tensor([[1.0, 2.0, 3.0, 10.0]])
    expected = (x - x.mean()) / (x.std(unbiased=True) + norm.eps)
    assert torch.allclose(norm(x), expected)
    assert not torch.allclose(norm(x), nn.LayerNorm(4, eps=norm.eps)(x))


def test_the_two_attention_masks_are_complementary():
    """One block sees only the past and its twin only the future — that pairing
    is the "bi-directional" in the model's name."""
    forward = causal_mask(4)[0, 0]
    backward = torch.transpose(causal_mask(4), 2, 3)[0, 0]
    for i in range(4):
        for j in range(4):
            assert torch.isinf(forward[i, j]) == (j > i)
            assert torch.isinf(backward[i, j]) == (j < i)


def test_timing_signal_is_sines_then_cosines():
    signal = timing_signal(TIMESTEP, 128)
    assert signal.shape == (1, TIMESTEP, 128)
    # Position 0 has every sine at 0 and every cosine at 1.
    assert torch.allclose(signal[0, 0, :64], torch.zeros(64), atol=1e-6)
    assert torch.allclose(signal[0, 0, 64:], torch.ones(64), atol=1e-6)


def test_load_published_state_dict_drops_the_unused_lstm():
    """The checkpoint carries weights for an LSTM its forward pass never calls."""
    model = BTCModel()
    published = dict(model.state_dict())
    published["output_layer.lstm.weight_ih_l0"] = torch.zeros(256, 128)
    model.load_published_state_dict(published)  # no error


def test_load_published_state_dict_still_rejects_a_wrong_checkpoint():
    """Dropping the LSTM keys must not turn into a blanket strict=False."""
    model = BTCModel()
    published = dict(model.state_dict())
    del published["self_attn_layers.embedding_proj.weight"]
    with pytest.raises(RuntimeError):
        model.load_published_state_dict(published)
