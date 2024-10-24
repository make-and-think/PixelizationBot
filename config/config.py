from yaml import load
from yaml import Loader


class Config:
    #TODO make dynaconf
    def __init__(self, data):
        self.__dict__.update(**data)


with open('config/config.yml') as f:
    config = load(f, Loader=Loader)
    config = Config(config)
