# UNet_GFN
CONFIG = {
    "runs": 5,
    "epochs": 200,
    "lr": 5e-3,
    "weight_decay": 5e-4,
    "n_hidden": 64,
    "n_layers": 5,
    "dropout": 0.0,
    "edgedrop": 0.2,
    "use_scale": True,
    "max_height": 5,
    "cluster_size_limit": 4,
    "pool_node_ratio": 0.3,
    "global_pool_methods": ["mean", "max"],  # Maybe att?
    "use_input_embedding": True,
}
