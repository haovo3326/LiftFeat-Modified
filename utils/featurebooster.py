from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def MLP(channels: List[int], do_bn: bool = False) -> nn.Module:
    """Multi-layer perceptron."""
    layers = []
    for i in range(1, len(channels)):
        layers.append(nn.Linear(channels[i - 1], channels[i]))
        if i < len(channels) - 1:
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def MLP_no_ReLU(channels: List[int], do_bn: bool = False) -> nn.Module:
    """Multi-layer perceptron without hidden ReLU activations."""
    layers = []
    for i in range(1, len(channels)):
        layers.append(nn.Linear(channels[i - 1], channels[i]))
        if i < len(channels) - 1 and do_bn:
            layers.append(nn.BatchNorm1d(channels[i]))
    return nn.Sequential(*layers)


class NormalEncoder(nn.Module):
    """Encoding of normal geometry using MLP."""
    def __init__(self, normal_dim: int, feature_dim: int, layers: List[int]) -> None:
        super().__init__()
        self.encoder = MLP_no_ReLU([normal_dim] + layers + [feature_dim])

    def forward(self, normals: torch.Tensor) -> torch.Tensor:
        return self.encoder(normals)


class DescriptorEncoder(nn.Module):
    """Encoding of visual descriptors using residual MLP."""
    def __init__(self, feature_dim: int, layers: List[int]) -> None:
        super().__init__()
        self.encoder = MLP([feature_dim] + layers + [feature_dim])

    def forward(self, desc: torch.Tensor) -> torch.Tensor:
        return desc + self.encoder(desc)


class AFTAttention(nn.Module):
    """Attention-free attention."""
    def __init__(self, d_model: int, n_heads: int = 1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        n_tokens, dim = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.view(n_tokens, self.n_heads, self.d_k).transpose(0, 1)
        k = k.view(n_tokens, self.n_heads, self.d_k).transpose(0, 1)
        v = v.view(n_tokens, self.n_heads, self.d_k).transpose(0, 1)

        k = torch.softmax(k, dim=-2)
        kv = (k * v).sum(dim=-2, keepdim=True)
        x = q * kv

        x = x.transpose(0, 1).reshape(n_tokens, dim)
        x = self.proj(x)
        return x + residual


class PositionwiseFeedForward(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.mlp = MLP([feature_dim, feature_dim * 2, feature_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


class AttentionalLayer(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int = 1):
        super().__init__()
        self.attn = AFTAttention(feature_dim, num_heads)
        self.ffn = PositionwiseFeedForward(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        return self.ffn(x)


class AttentionalNN(nn.Module):
    def __init__(self, feature_dim: int, layer_num: int, num_heads: int = 1) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalLayer(feature_dim, num_heads)
            for _ in range(layer_num)
        ])

    def forward(self, desc: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            desc = layer(desc)
        return desc


class PointwiseProjection(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.proj = nn.Conv1d(input_dim, output_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.t().unsqueeze(0)
        x = self.proj(x)
        return x.squeeze(0).t()


class FeatureProjection(nn.Module):
    """Project concatenated descriptor and normal features back to descriptor space."""
    def __init__(self, input_dim: int, output_dim: int, layers: List[int]):
        super().__init__()
        self.mlp = MLP([input_dim] + layers + [output_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class FeatureBooster(nn.Module):
    default_config = {
        "modified": False,
        "descriptor_dim": 128,
        "normal_dim": 192,
        "normal_encoder": [32, 64, 128],
        "descriptor_encoder": [64, 64],
        "feature_projection": [128, 64, 64],
        "num_heads": 1,
        "Attentional_layers": 3,
        "last_activation": "relu",
        "l2_normalization": True,
        "output_dim": 128,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}
        self.modified = self.config.get("modified") is True

        if self.modified:
            self._init_modified()
        else:
            self._init_original()

        self.last_activation = self._make_activation(self.config.get("last_activation"))

    def _init_modified(self) -> None:
        descriptor_dim = self.config["descriptor_dim"]
        self.normal_proj = PointwiseProjection(self.config["normal_dim"], descriptor_dim)
        self.feat_project = FeatureProjection(
            input_dim=descriptor_dim * 2,
            output_dim=descriptor_dim,
            layers=self.config["feature_projection"],
        )
        self.attn_proj = AttentionalNN(
            feature_dim=descriptor_dim,
            layer_num=self.config["Attentional_layers"],
            num_heads=self.config["num_heads"],
        )

    def _init_original(self) -> None:
        descriptor_dim = self.config["descriptor_dim"]
        self.nenc = NormalEncoder(
            self.config["normal_dim"],
            descriptor_dim,
            self.config["normal_encoder"],
        )
        self.denc = DescriptorEncoder(descriptor_dim, self.config["descriptor_encoder"])
        self.attn_proj = AttentionalNN(
            feature_dim=descriptor_dim,
            layer_num=self.config["Attentional_layers"],
            num_heads=self.config["num_heads"],
        )

    @staticmethod
    def _make_activation(activation):
        if not activation:
            return None
        activation = activation.lower()
        if activation == "relu":
            return nn.ReLU()
        if activation == "sigmoid":
            return nn.Sigmoid()
        if activation == "tanh":
            return nn.Tanh()
        raise ValueError(f'Not supported activation "{activation}".')

    """
    3D GFL Ablation Study
    Variations          Fusion                      Attention Head      Residual    In Training
    GFL0 (LiftFeat)     2x MLP                      Single              No                    
    GFL1                EMT + MLP                   Single              No          
    GFL2                2x MLP                      Multi               No          
    GFL3                2x MLP                      Single              Yes         X
    GFL4 (Aggregated)   EMT + MLP                   Multi               Yes         
    """
    def forward(self, desc: torch.Tensor, *inputs: torch.Tensor) -> torch.Tensor:
        if self.modified:
            if len(inputs) != 1:
                raise TypeError("Modified FeatureBooster expects forward(desc, normals).")
            desc = self._forward_modified(desc, inputs[0])
        else:
            if len(inputs) != 1:
                raise TypeError("Original FeatureBooster expects forward(desc, normals).")
            desc = self._forward_original(desc, inputs[0])

        if self.last_activation is not None:
            desc = self.last_activation(desc)
        if self.config["l2_normalization"]:
            desc = F.normalize(desc, dim=-1)
        return desc

    # def _forward_modified(self, desc: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    #     residual = desc
    #     normals = self.normal_proj(normals)
    #     desc = desc * torch.tanh(normals)
    #     desc = self.feat_project(desc)
    #     desc = self.attn_proj(desc)
    #     return desc

    def _forward_original(
        self,
        desc: torch.Tensor,
        normals: torch.Tensor,
    ) -> torch.Tensor:
        residual = desc
        desc = self.denc(desc)
        desc = desc + self.nenc(normals)
        desc = self.attn_proj(desc)
        return desc + residual
