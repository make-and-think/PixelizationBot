from yaml import load
from yaml import Loader

class Config:
  def __init__(self, data):
    self.__dict__.update(**data)

with open('config.yml') as f:
  config = load(f, Loader=Loader)
  config = Config(config)
