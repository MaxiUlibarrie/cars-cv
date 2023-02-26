import json
import os
from types import SimpleNamespace
from logger import Logger

logger = Logger(os.environ.get('LOG_OUTPUT'))

class Config():

    __shared_instance = None

    def __new__(cls):
        if cls.__shared_instance is None:
            cls.__shared_instance = super().__new__(cls)
            cls.load_config()
        
        return cls.__shared_instance  

    @classmethod
    def load_config(cls):

        CONFIG_PATH_FILE = os.environ.get("CONFIG_PATH_FILE")

        with open(CONFIG_PATH_FILE) as config_file:
            config_json = config_file.read()

        cls.get = json.loads(config_json, object_hook=lambda d: SimpleNamespace(**d))

        logger.log_L2("Config loaded")
        