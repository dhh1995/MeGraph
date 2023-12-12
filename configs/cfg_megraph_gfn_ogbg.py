# Megraph GFN for ogbg
CONFIG = {
    "runs": 5,
    "lr": 0.001,
    "weight_decay": 5e-4,
    "e_hidden": None,
    "n_hidden": 300,
    "n_layers": 4,
    "activation": "relu",
    "dropout": 0.5,
    "norm_layer": "layer",
    "epochs": 100,
    "use_input_embedding": True,
    "add_reverse_edges": False,
    "self_loop": True,
    "to_simple_graph": False,
}
