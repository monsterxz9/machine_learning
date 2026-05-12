"""
Smoke test:用合成数据验证 PyTorch pipeline 通畅,不依赖真实赛题数据。
"""

import os
import sys
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import FourierFeatures, MicrostripMLP


def test_fourier_output_shape():
    ff = FourierFeatures(num_features=8)
    out = ff(torch.randn(32))
    assert out.shape == (32, 16)


def test_fourier_bounded():
    ff = FourierFeatures(num_features=4)
    out = ff(torch.randn(100) * 5)
    assert out.min().item() >= -1.0001
    assert out.max().item() <= 1.0001


def test_model_forward_shape():
    model = MicrostripMLP(geo_dim=4, num_fourier=8, hidden_layers=[32, 32])
    out = model(torch.randn(16, 4), torch.randn(16))
    assert out.shape == (16, 4)


def test_model_backward_has_grad():
    model = MicrostripMLP(geo_dim=4, num_fourier=8, hidden_layers=[32, 32])
    out = model(torch.randn(16, 4), torch.randn(16))
    loss = nn.functional.mse_loss(out, torch.randn(16, 4))
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.parameters()
    )
    assert has_grad


def test_train_step_reduces_loss():
    """30 步 Adam 后 loss 应该明显下降。"""
    torch.manual_seed(0)
    model = MicrostripMLP(geo_dim=4, num_fourier=8, hidden_layers=[64, 64])
    geo = torch.randn(64, 4)
    freq = torch.randn(64)
    target = torch.randn(64, 4)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    with torch.no_grad():
        initial_loss = nn.functional.mse_loss(model(geo, freq), target).item()

    for _ in range(30):
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(geo, freq), target)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = nn.functional.mse_loss(model(geo, freq), target).item()
    assert final_loss < initial_loss * 0.5, f"loss {initial_loss:.4f} -> {final_loss:.4f}"


def test_save_load_roundtrip():
    """save → load 后预测一致。"""
    torch.manual_seed(0)
    cfg = {"geo_dim": 4, "num_fourier": 8, "hidden_layers": [32, 32]}
    model = MicrostripMLP(**cfg)
    model.eval()

    geo = torch.randn(8, 4)
    freq = torch.randn(8)
    with torch.no_grad():
        before = model(geo, freq).clone()

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt.pt")
        torch.save({"model_state": model.state_dict(), "config": cfg}, path)

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model2 = MicrostripMLP(**ckpt["config"])
        model2.load_state_dict(ckpt["model_state"])
        model2.eval()

        with torch.no_grad():
            after = model2(geo, freq)

    assert torch.allclose(before, after, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
