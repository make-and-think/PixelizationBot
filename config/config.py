import logging

from yaml import load
from yaml import Loader
import os
import torch
import torch.multiprocessing as mp
import argparse
parser = argparse.ArgumentParser(description="PixelizationBot")
parser.add_argument('--config', type=str, required=False, help='Path to the config file')
args = parser.parse_args()
CONFIG_FILE_PATH = 'config/config.yml' if not args.config else args.config

class Config:
    # TODO make dynaconf
    def __init__(self, data):
        self.__dict__.update(**data)

    def get(self, name: str):
        return self.__dict__.get(name)


with open(CONFIG_FILE_PATH) as f:
    config = load(f, Loader=Loader)
    config = Config(config)

    # force disable cuda and set limit to one thread

    mp.set_start_method('spawn', force=True)
    if config.get("NUM_TORCH_THREADS"):
        torch.set_num_threads(config.NUM_TORCH_THREADS)
    if config.get("FORCE_USE_CPU"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.set_num_threads(1)



if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('logs/bot.log'),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)
