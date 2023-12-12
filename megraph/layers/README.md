
This folder includes implementations of our **Mee Layer** and other fundamental layers. 

The ```graph_layers``` directory contains fundamental layers such as GCN, GAT, SAGE, PAN, and GFN. To call these layers, specify the layer name using the `--layer` option, e.g., use `--layer gfn` to use the GFN layer.


The ```attention.py``` file includes the **AttentionWeightLayer**  for cross-updating with attention in the inter-graph.

The  ```conv_block.py``` file contains the fundamental implementation of a GNN block that processes graph data using the layer specified via `--layer`.

The ```encoder.py``` file contains implementations of embedding methods for input data from various graph datasets.


The ```mee.py``` file contains the implementation of our **Mee Layer**.


The ```mlp.py``` file ontains the implementation of an MLP (multilayer perceptron) layer, which is typically used as the output layer in a model.