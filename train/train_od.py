import subprocess as sp
import argparse
import os

# config
from useful import Config

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
        default=config.epochs,
        help="number of epochs"
    )

    parser.add_argument(
        "--batch-size",
        type=str,
        required=False,
        default=config.batch_size,
        help="BATCH_SIZE for training"
    )

    parser.add_argument(
        "--workers",
        type=str,
        required=False,
        default=config.workers,
        help="BATCH_SIZE for training"
    )

    args, _ = parser.parse_known_args()
    
    print("### ARGUMENTS ###")
    print(args)

    return args

def train_od(config):
    opt = get_opts(config)
    version_model = opt.version_model.zfill(2)
    path_train_od = os.environ.get("PATH_TRAIN_OD")
    sp.run(["/bin/bash", "-m", path_train_od, "-v", version_model,"-e", opt.epochs, "-b", opt.batch_size, "-w", opt.workers])

if __name__ == "__main__":
    config = Config()
    train_od(config)