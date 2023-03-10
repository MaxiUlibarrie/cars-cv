import argparse
import os
from od_models import Yolov5
from pc_model import PreventCollision
from video_writer import VideoWriter

from config import Config
from logger import Logger

config = Config()
logger = Logger()

def get_opts(config):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        default='0',
        help="Version of OD model - weights."
    )

    parser.add_argument(
        "--width",
        type=int,
        required=False,
        default=config.get.detection.width,
        help="Image width."
    )

    parser.add_argument(
        "--height",
        type=int,
        required=False,
        default=config.get.detection.height,
        help="Image height."
    )

    parser.add_argument(
        "--conf-thres",
        type=str,
        required=False,
        default=config.get.detection.conf_thres,
        help="Confidence threshold of OD model."
    )

    parser.add_argument(
        "--iop-thres",
        type=str,
        required=False,
        default=config.get.detection.iop_thres,
        help="Intersection over person for detecting people in danger."
    )

    parser.add_argument(
        "--format-video",
        type=str,
        required=False,
        default="mp4",
        help="Format of videos (input and output)."
    )

    parser.add_argument(
        "--video-codec",
        type=str,
        required=False,
        default="mp4v",
        help="Format codec of output video."
    )

    args, _ = parser.parse_known_args()
    
    logger.log_L1("ARGUMENTS")
    logger.log_L2(vars(args))

    return args

if __name__ == "__main__":
    
    try: 
        opt = get_opts(config)
        yolo = Yolov5(opt.weights)
        pc = PreventCollision(yolo)
        vw = VideoWriter(pc)

        format_video = opt.format_video
        video_codec = opt.video_codec

        video_files = os.listdir(os.environ.get('VIDEO_PATH'))
        video_path_file = f"{os.environ.get('VIDEO_PATH')}/{video_files[0]}"

        widht = int(opt.width)
        height = int(opt.height)
        conf_thres = float(opt.conf_thres)
        iop_thres = float(opt.iop_thres)

        vw.generate_video(
            video_path = video_path_file, 
            width = widht, 
            height = height,
            conf_thres = conf_thres,
            iop_thres = iop_thres,
            format_video = format_video,
            video_codec = video_codec)
    except Exception as e:
        logger.log_error(e)
    