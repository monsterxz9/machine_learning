"""模型定义:Fourier 频率编码 + MLP。"""

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """
    把标量频率编码成 2*num_features 维的 sin/cos 多尺度特征。

    使用几何递增的频率 scale 覆盖多尺度震荡,缓解 ReLU/GELU 网络对
    高频信号的 spectral bias(优先学低频)。
    """

    def __init__(self, num_features: int = 16, base_freq: float = 1.0) -> None:
        super().__init__()
        scales = base_freq * (2.0 ** torch.arange(num_features, dtype=torch.float32))
        self.register_buffer("scales", scales)

    def forward(self, freq: torch.Tensor) -> torch.Tensor:
        if freq.dim() == 1:
            freq = freq.unsqueeze(-1)
        scaled = freq * self.scales
        return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)


class MicrostripMLP(nn.Module):
    """
    输入:几何 (B, geo_dim) + 频率 (B,)  → Fourier 编码后拼接
    输出:(B, 4) = [S11.real, S11.imag, S12.real, S12.imag]
    """

    def __init__(
        self,
        geo_dim: int = 4,
        num_fourier: int = 16,
        hidden_layers: list[int] | None = None,
    ) -> None:
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [256, 256, 128, 128, 64]

        self.geo_dim = geo_dim
        self.num_fourier = num_fourier
        self.hidden_layers = list(hidden_layers)

        self.fourier = FourierFeatures(num_features=num_fourier)
        input_dim = geo_dim + 2 * num_fourier

        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.GELU())
            prev = h
        layers.append(nn.Linear(prev, 4))
        self.net = nn.Sequential(*layers)

    def forward(self, geo: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        x = torch.cat([geo, self.fourier(freq)], dim=-1)
        return self.net(x)

    def export_config(self) -> dict:
        return {
            "geo_dim": self.geo_dim,
            "num_fourier": self.num_fourier,
            "hidden_layers": self.hidden_layers,
        }
