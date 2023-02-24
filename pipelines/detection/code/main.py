import argparse
import os

from config import Config
from logger import Logger

from od_models import Yolov5
from pc_model import PreventCollision
from video_writer import VideoWriter

logger = Logger()
config = Config()

def get_opts(config):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        default='0',
        help="Version of od model - weights."
    )

    parser.add_argument(
        "--width",
        type=int,
        required=False,
        default=640,
        help="Image width."
    )

    parser.add_argument(
        "--height",
        type=int,
        required=False,
        default=480,
        help="Image height."
    )

    parser.add_argument(
        "--conf-thres",
        type=str,
        required=False,
        default="0.3",
        help="Confidence threshold of OD model."
    )

    parser.add_argument(
        "--iop-thres",
        type=str,
        required=False,
        default="0.5",
        help="Intersection over person for detecting people in danger."
    )

    args, _ = parser.parse_known_args()
    
    print("### ARGUMENTS ###")
    print(vars(args))

    return args

if __name__ == "__main__":
    opt = get_opts(config)
    yolo = Yolov5(opt.weights)
    pc = PreventCollision(yolo)
    vw = VideoWriter(pc)

    video_files = os.listdir(os.environ.get('PATH_VIDEO'))
    video_path_file = f"{os.environ.get('PATH_VIDEO')}/{video_files[0]}"

    output_path_file = f"{os.environ.get('PATH_OUTPUT')}/{video_files[0]}_out"

    widht = int(opt.width)
    height = int(opt.height)
    conf_thres = float(opt.conf_thres)
    iop_thres = float(opt.iop_thres)

    logger.log_1("Started to generate video.")
    vw.generate_video(
        video_path = video_path_file, 
        video_output = output_path_file,
        width = widht, 
        height = height,
        conf_thres = conf_thres,
        iop_thres = iop_thres)
    