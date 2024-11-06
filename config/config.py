from yaml import load
from yaml import Loader
import os


class Config:
    # TODO make dynaconf
    def __init__(self, data):
        self.__dict__.update(**data)


with open('config/config.yml') as f:
    config = load(f, Loader=Loader)
    config = Config(config)

    # force disable cuda and set limit to one thread
    if config.__dict__.get("FORCE_USE_CPU"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if config.__dict__.get("FORCE_USE_CPU"):
        import torch
        torch.set_num_threads(1)
