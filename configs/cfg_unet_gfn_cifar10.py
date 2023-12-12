# UNet GFN for cifar10
CONFIG = {
    "runs": 4,
    "drop_last": False,
    "lr_scheduler": "multiste",
    "lr_scheduler_indicator": "loss",
    "reduce_factor": 0.5,
    "reduce_patience": 5,
    "milestones": [60],
    "min_lr": 1e-5,
    "lr": 0.001,
    "weight_decay": 0.0,
    "e_hidden": None,
    "n_hidden": 144,
    "n_layers": 3,
    "activation": "relu",
    "dropout": 0.3,
    "edgedrop": 0.2,
    "norm_layer": "layer",
    "epochs": 100,
    "train_batch_size": 128,
    "eval_batch_size": 128,
    "stem_beta": 1,
    "branch_beta": 1,  # default: 0.5
    "last_hidden_dims": [72, 36],
    "last_simple": True,
    "use_input_embedding": True,
    "max_height": 5,
    "pool_aggr_edge": ["sum"],
}
