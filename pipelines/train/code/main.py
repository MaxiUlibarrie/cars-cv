import subprocess as sp
import argparse
import os

from config import Config
from logger import Logger

logger = Logger()
config = Config()

def get_opts(config):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--version-model",
        type=str,
        required=True,
        default='0',
        help="Version of model"
    )

    parser.add_argument(
        "--epochs",
        type=str,
        required=False,
        default=config.get.train.epochs,
        help="number of epochs"
    )

    parser.add_argument(
        "--batch-size",
        type=str,
        required=False,
        default=config.get.train.batch_size,
        help="batch size for training"
    )

    parser.add_argument(
        "--workers",
        type=str,
        required=False,
        default=config.get.train.workers,
        help="number of workers"
    )

    parser.add_argument(
        "--yolo-weights",
        type=str,
        required=False,
        default=config.get.train.yolo_weights,
        choices=[
            "yolov5n",
            "yolov5s",
            "yolov5m", 
            "yolov5l",
            "yolov5x"
        ],
        help="YOLOv5 weights"
    )

    parser.add_argument(
        "--image-size",
        type=str,
        required=False,
        default=config.get.train.image_size,
        help="image size for training"
    )

    args, _ = parser.parse_known_args()
    
    logger.log_L1("ARGUMENTS")
    logger.log_L2(vars(args))

    return args

if __name__ == "__main__":
    opt = get_opts(config)
    version_model = opt.version_model.zfill(2)
    path_train_od_sh = os.environ.get("TRAIN_OD_SH_FILE")
    logger.log_L1("Started to train OD model.")
    sp.run([
        "/bin/bash", "-m", path_train_od_sh, 
        "-v", version_model,
        "-e", opt.epochs, 
        "-b", opt.batch_size, 
        "-w", opt.workers, 
        "-y", opt.yolo_weights,
        "-i", opt.image_size
    ])
    logger.log_L1("Finished training OD model.")
