"""
config.py
Written by Li Jiang
"""

import argparse

import yaml


def get_parser():
    parser = argparse.ArgumentParser(description="Point Cloud Segmentation")
    parser.add_argument(
        "--config", type=str, default="config/pointgroup_default_scannet.yaml", help="path to config file"
    )

    # pretrain
    parser.add_argument("--pretrain", type=str, default=None, help="path to pretrain model")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--threshold_ins", type=float, default=0.5)
    parser.add_argument("--min_pts_num", type=int, default=50)

    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--train", default=True)

    args_cfg = parser.parse_args()
    assert args_cfg.config is not None
    with open(args_cfg.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args_cfg, k, v)

    return args_cfg


cfg = get_parser()

setattr(cfg, "exp_path", cfg.output_path)
