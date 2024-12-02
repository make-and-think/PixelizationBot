from yaml import load
from yaml import Loader
import os
import torch
import torch.multiprocessing as mp


class Config:
    # TODO make dynaconf
    def __init__(self, data):
        self.__dict__.update(**data)

    def get(self, name: str):
        return self.__dict__.get(name)


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
