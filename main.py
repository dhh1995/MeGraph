import os
import sys

sys.path.append(os.getcwd())

import argparse
import traceback

import numpy as np
import taichi as ti
from megraph.args_utils import get_args_and_model
from megraph.datasets import DotDict, graph_dataset_manager
from megraph.layers import layer_factory, register_layers_args
from megraph.logger import logger
from megraph.models import model_factory, register_models_args
from megraph.torch_utils import get_num_params, set_global_seed
from megraph.trainer import Trainer
from megraph.utils import register_logging_args, set_logger

parser = argparse.ArgumentParser(description="Node classification on citation networks")
register_logging_args(parser)
graph_dataset_manager.register_dataset_args(parser)
register_layers_args(parser)
register_models_args(parser)
Trainer.register_trainer_args(parser)
parser.add_argument("--seed", "-se", type=int, default=2022)
parser.add_argument(
    "--configs-dir", "-cd", type=str, default="configs", help="configs dir"
)
parser.add_argument(
    "--config-file", "-cfg", type=str, default=None, help="config filename"
)
parser.add_argument("--runs", "-rn", type=int, default=10, help="number of runs")
parser.add_argument("--record-graphs", "-rg", action="store_true", help="record graphs")
parser.add_argument("--save-model", "-sm", action="store_true", help="save model")
parser.add_argument(
    "--load-model-path", "-lmp", type=str, default=None, help="load model"
)
parser.add_argument("--debug", "-de", action="store_true", help="debug")
parser.add_argument("--gpu-id", "-gid", type=int, default=0, help="gpu id")

args, graph_layer, graph_model = get_args_and_model(
    parser, layer_factory, model_factory
)
set_global_seed(args)
dump_dir = set_logger(args)

record_graphs_dir = None
if args.record_graphs:
    if args.model == "megraph":
        record_graphs_dir = os.path.join(dump_dir, "graphs")
    else:
        args.record_graphs = False
        logger.info("Only megraph model need record graphs")

save_model_dir = None
if args.save_model:
    save_model_dir = os.path.join(dump_dir, "models")
    os.mkdir(save_model_dir)

ti.init(random_seed=args.seed)
# ti.init(arch=ti.gpu)


def run(run_id):
    logger.info(f"cmd: {args.raw_cmdline}")
    task = graph_dataset_manager.task
    dataset, meta_data = graph_dataset_manager.get_dataset_and_meta_data()
    masks = graph_dataset_manager.get_dataset_split(run_id)
    input_dims, output_dims, pe_dim = graph_dataset_manager.get_input_output_dim()
    if args.layer in ["gcn"]:
        input_dims[2] = 0
    # create graph model
    if args.model in ["plain", "megraph", "unet", "hgnet"]:

        def build_conv(**kwargs):
            return graph_layer.from_args(args, **kwargs)

        model = graph_model.from_args(
            args,
            input_dims=input_dims,
            output_dims=output_dims,
            pe_dim=pe_dim,
            task=task,
            embed_method=meta_data.get("embed_method", {}),
            build_conv=build_conv,
        )
    else:
        in_feats = input_dims[1]
        n_classes = graph_dataset_manager.get_num_classes()
        model = graph_model.from_args(args, in_feats=in_feats, n_classes=n_classes)

    logger.info(model)
    logger.info(f"Num params of {args.model}: {get_num_params(model)}")

    loss_function = graph_dataset_manager.get_loss_function()
    evaluator = graph_dataset_manager.get_evaluator()
    trainer = Trainer.from_args(
        args,
        gpu_id=args.gpu_id,
        model=model,
        task=task,
        loss_function=loss_function,
        evaluator=evaluator,
        meta_data=meta_data,
        record_graphs_dir=record_graphs_dir if run_id == 0 else None,  # only first run
        save_model_dir=save_model_dir,
    )
    if args.load_model_path is not None:
        trainer.load(args.load_model_path)
    return trainer.train(args.epochs, run_id, dataset, masks)


def main():
    graph_dataset_manager.set_params_from_args(args)
    # load and preprocess dataset
    graph_dataset_manager.load_and_process_dataset()
    all_runs = []
    for i in range(args.runs):
        all_runs.append(run(i))
        mean_acc = np.mean(all_runs)
        std_acc = np.std(all_runs)
        acc_str = ", ".join([f"{acc :.4f}" for acc in all_runs])
        logger.info(f"All runs: [{acc_str}]")
        logger.info(f"Average Test Accuracy: {mean_acc :.4f}(mean) {std_acc: .4f}(std)")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        error_info = traceback.format_exc()
        logger.error(f"\n{error_info}")
