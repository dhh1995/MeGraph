This folder contains the [`configs`](../configs) used in our paper to reproduce the results of our experiments.


The configuration will be utilized based on your specified parameters. For instance, when using `-md megraph -ly gfn -dname zinc`, the configuration file 'configs/cfg_hgnet_gfn_zinc' will be used. It is important to note that any configurations set through input commands will overwrite the settings specified in the configuration file.

Moreover, you have the option to use your customized configuration file by setting `-cfg`. For example, to use our [`best`](../configs/best) configuration, you can set `-cfg configs/best/cfg_megraph_gfn_zinc.py`.