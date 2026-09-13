from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def MLP(channels: List[int], do_bn: bool = False) -> nn.Module:
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Linear(channels[i - 1], channels[i]))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class AFTAttention(nn.Module):
    """ Attention-free attention """
    def __init__(self, d_model: int, n_heads) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        N, D = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # [N, D] -> [N, H, d_k]
        q = q.view(N, self.n_heads, self.d_k)
        k = k.view(N, self.n_heads, self.d_k)
        v = v.view(N, self.n_heads, self.d_k)

        # [N, H, d_k] -> [H, N, d_k]
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        # normalize keys over tokens
        k = torch.softmax(k, dim=-2)

        # [H, N, d_k] -> [H, 1, d_k]
        kv = (k * v).sum(dim=-2, keepdim=True)

        # [H, N, d_k]
        x = q * kv

        # concatenate heads
        # [H, N, d_k] -> [N, H, d_k]
        x = x.transpose(0, 1)

        # [N, H, d_k] -> [N, D]
        x = x.reshape(N, D)

        x = self.proj(x)

        # residual
        x = x + residual

        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.mlp = MLP([feature_dim, feature_dim*2, feature_dim])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.mlp(x)
        x += residual
        return x


class AttentionalLayer(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = AFTAttention(feature_dim, num_heads)
        self.ffn = PositionwiseFeedForward(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        x = self.ffn(x)
        return x


class AttentionalNN(nn.Module):
    def __init__(self, feature_dim: int, num_heads, layer_num: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalLayer(feature_dim, num_heads)
            for _ in range(layer_num)])

    def forward(self, desc: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            desc = layer(desc)
        return desc

class PointwiseProjection(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.proj = nn.Conv1d(input_dim, output_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pointwise 1x1 projection for token features shaped as (N, C).
        x = x.t().unsqueeze(0)
        x = self.proj(x)
        return x.squeeze(0).t()

class FeatureProjection(nn.Module):
    """
    Project the 64-dimensional attended GFL feature back to descriptor space.
    """
    def __init__(self, input_dim: int, output_dim: int, layers: List[int]):
        super().__init__()
        self.mlp = MLP([input_dim] + layers + [output_dim])

    def forward(self, x):
        return self.mlp(x)


class FeatureBooster(nn.Module):
    default_config = {
        'descriptor_dim': 128,
        'normal_dim': 192,
        'feature_projection': [128, 64, 64],
        'Attentional_layers': 3,
        'last_activation': 'relu',
        'l2_normalization': True,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.desc_proj = PointwiseProjection(self.config['descriptor_dim'], self.config['descriptor_dim'])
        self.normal_proj = PointwiseProjection(self.config['normal_dim'], self.config['descriptor_dim'])


        self.feat_project = FeatureProjection(
            input_dim=self.config['descriptor_dim'] * 2,
            output_dim=self.config['descriptor_dim'],
            layers=self.config['feature_projection'],
        )

        self.attention_dim = self.config['descriptor_dim']

        self.attn_proj = AttentionalNN(
            feature_dim=self.attention_dim,
            num_heads= self.config['num_heads'],
            layer_num=self.config['Attentional_layers']
        )

        if self.config.get('last_activation', False):
            if self.config['last_activation'].lower() == 'relu':
                self.last_activation = nn.ReLU()
            elif self.config['last_activation'].lower() == 'sigmoid':
                self.last_activation = nn.Sigmoid()
            elif self.config['last_activation'].lower() == 'tanh':
                self.last_activation = nn.Tanh()
            else:
                raise Exception('Not supported activation "%s".' % self.config['last_activation'])
        else:
            self.last_activation = None

    """
    Architectural Ablation Study
    Variant         Fusion                      Attention Head      Residual    
    M1*             1x1 Conv + Concat + MLP     Single              No          
    M2              1x1 Conv + Concat + MLP     Single              No
    M3              1x1 Conv + Concat + MLP     Multi               No
    M4              1x1 Conv + Concat + MLP     Multi               Yes     
    """
    def forward(self, desc, normals):
        residual = desc
        desc = self.desc_proj(desc)                         # raw desc -> 1x1 Conv -> new desc
        normals = self.normal_proj(normals)                 # raw normals -> 1x1 Conv -> new normals
        desc = torch.cat([desc, normals], dim = -1) # Concatenation: [desc: normals]
        desc = self.feat_project(desc)                      # MLP projection
        desc = self.attn_proj(desc)                         # Multi-head/Single-head attention
        desc = desc + residual

        if self.last_activation is not None:
            desc = self.last_activation(desc)
        # L2 normalization
        if self.config['l2_normalization']:
            desc = F.normalize(desc, dim=-1)

        return desc

if __name__ == "__main__":
    from config import featureboost_config
    fb_net = FeatureBooster(featureboost_config)

    descs=torch.randn([1900,64])
    normals=torch.randn([1900,192])

    import pdb;pdb.set_trace()

    descs_refine=fb_net(descs,normals)

    print(descs_refine.shape)
