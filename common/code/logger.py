import logging 
import sys
import os

ENV_VAR_LOG_OUTPUT = "LOG_OUTPUT"

class Logger():

    __shared_instance = None

    def __new__(cls):
        if cls.__shared_instance is None:
            path_log = os.environ.get(ENV_VAR_LOG_OUTPUT)
            cls.logger = cls.setup_custom_logger("PCS", path_log)
            cls.__shared_instance = super().__new__(cls)
        
        return cls.__shared_instance
        
    @classmethod
    def setup_custom_logger(cls, name, path_log):
        formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S')
        handler = logging.FileHandler(path_log, mode='w')
        handler.setFormatter(formatter)
        screen_handler = logging.StreamHandler(stream=sys.stdout)
        screen_handler.setFormatter(formatter)
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        logger.addHandler(screen_handler)
        return logger

    @classmethod
    def log_L1(cls, txt):
        cls.logger.info(f"##### {txt} #####")

    @classmethod
    def log_L2(cls, txt):
        cls.logger.info(f"# {txt} #")

    @classmethod
    def log_L3(cls, txt):
        cls.logger.info(f"-/{txt}/-")

    @classmethod
    def log_error(cls, txt):
        cls.logger.error(f"##### {txt} #####")
