from yaml import load
from yaml import Loader
import os
import torch
import torch.multiprocessing as mp
import argparse
import logging

logger = logging.getLogger(__name__)


class Config:
    # TODO make dynaconf
    def __init__(self, data):
        self.__dict__.update(**data)

    def get(self, name: str):
        return self.__dict__.get(name)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the bot with specified device.")
    parser.add_argument('--device', choices=['cuda', 'cpu'], help="Specify the device to use: 'cuda' or 'cpu'.")
    return parser.parse_args()


def configure_device(device_choice, config):
    if device_choice == 'cpu' or (device_choice is None and config.get("FORCE_USE_CPU")):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.set_num_threads(1)
        logger.info("Using device: cpu")
    elif device_choice == 'cuda':
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)  # Allow CUDA usage
        logger.info("Using device: cuda")


with open('config/config.yml') as f:
    config = load(f, Loader=Loader)
    config = Config(config)

    # force disable cuda and set limit to one thread

    mp.set_start_method('spawn', force=True)
    if config.get("NUM_TORCH_THREADS"):
        torch.set_num_threads(config.NUM_TORCH_THREADS)
    if config.get("FORCE_USE_CPU"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.set_num_threads(1)
