# modified_fusion_featureboost_config = {
#     "modified": True,
#     "normal_dim": 192,
#     "normal_encoder": [128, 64, 64],
#     "feature_projection": [64, 64],
#     "descriptor_dim": 64,
#     "num_heads": 4,
#     "Attentional_layers": 3,
#     "last_activation": None,
#     "l2_normalization": False,
# }

original_fusion_featureboost_config = {
    "modified": False,
    "normal_dim": 192,
    "normal_encoder": [128, 64, 64],
    "descriptor_encoder": [64, 64],
    "descriptor_dim": 64,
    "num_heads": 1,
    "Attentional_layers": 3,
    "last_activation": None,
    "l2_normalization": False,
    "output_dim": 64,
}
