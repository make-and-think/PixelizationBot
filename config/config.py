from yaml import load
from yaml import Loader
import os

class Config:
    #TODO make dynaconf
    def __init__(self, data):
        self.__dict__.update(**data)


with open('config/config.yml') as f:
    config = load(f, Loader=Loader)
    config = Config(config)
    if config.__dict__.get("force_use_cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if hasattr(config, 'NUM_THREADS'):
        import torch
        torch.set_num_threads(config.NUM_THREADS)
