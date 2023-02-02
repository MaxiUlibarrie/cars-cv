import json
import os

class Config():

    def __init__(self):

        CONFIG_PATH_FILE = os.environ.get("CONFIG_PATH_FILE")

        with open(CONFIG_PATH_FILE) as config_file:
            config_json = config_file.read()

        config = json.loads(config_json)

        self.batch_size = str(config["model"]["batch-size"])
        self.epochs = str(config["model"]["epochs"])
        self.workers = str(config["model"]["workers"])
        self.yolo_weights = str(config["model"]["yolo_weights"])
        self.image_size = str(config["model"]["image_size"])

        print("### Config loaded ###")
        