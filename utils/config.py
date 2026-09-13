import os
import sys
import numpy as np

featureboost_config = {
    "normal_dim": 192,
    "feature_projection": [128, 64, 64],
    "descriptor_dim": 64,
    "num_heads": 4,
    "Attentional_layers": 3,
    "last_activation": None,
    "l2_normalization": None,
}
