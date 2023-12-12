
This folder includes implementations of our **Megraph** model as well as models for other baselines.

The files ```megraph.py```, ```hgnet.py``` and ```unet.py``` are the implementations of **Megraph**, **HGNET[<sup>1</sup>](#refer-anchor-1)** and **Graph U-Net[<sup>2</sup>](#refer-anchor-2)**.

The ```edgepool.py``` file will process graph datas with the pool operator.

The ```models.py``` file provides the implementation of other base models, such as GCN and GAT.

The ```model.py``` file contains the backbone of all models. To create your own model, simply inherit from this class and build upon it.

The ```plain.py``` file contains the backbone of models without pool operator.



<div id="refer-anchor-1"></div>

- [1] Rampášek L, Wolf G. Hierarchical graph neural nets can capture long-range interactions[C]//2021 IEEE 31st International Workshop on Machine Learning for Signal Processing (MLSP). IEEE, 2021: 1-6.


<div id="refer-anchor-2"></div>

- [2] Gao H, Ji S. Graph u-nets[C]//international conference on machine learning. PMLR, 2019: 2083-2092.