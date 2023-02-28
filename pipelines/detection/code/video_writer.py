import cv2
import numpy as np
from tqdm import tqdm
import os
from dateutil.relativedelta import relativedelta

from logger import Logger

logger = Logger()

class VideoWriter():
    
    color_map = {
        "person": (255, 0, 0), # blue
        "bicycle": (255, 255, 0), # cyan 
        "car": (0, 128, 255), # orange 
        "motorcycle": (255, 0, 128), # purple 
        "bus": (255, 0, 255), # magenta
        "collision_zone": (0, 255, 255), # yellow
        "person_in_danger": (0, 0, 255) # red
    }
    
    def __init__(self, pc_model):
        self.pc_model = pc_model 

    def generate_video(self, video_path,
                       width = 640, 
                       height = 480,
                       conf_thres = 0.3,
                       iop_thres = 0.5,
                       format_video = "mp4",
                       video_codec = "mp4v"):
        """
        Generates the full overlay output and alert output from the input video. 
        """
        video_files = os.listdir(os.environ.get('VIDEO_PATH'))
        output_full = f"{video_files[0].replace('.' + format_video,'')}_full.{format_video}"
        output_full_path = f"{os.environ.get('OUTPUT_PATH')}/{output_full}"

        output_alerts = f"{video_files[0].replace('.' + format_video,'')}_alerts.{format_video}"
        output_alerts_path = f"{os.environ.get('OUTPUT_PATH')}/{output_alerts}"
        
        logger.log_L1("Started to generate video.")
        cap = cv2.VideoCapture(video_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        out_full = cv2.VideoWriter(output_full_path, 
                                   cv2.VideoWriter_fourcc(*video_codec),
                                   fps,
                                   (width, height))
        
        out_alerts = cv2.VideoWriter(output_alerts_path, 
                                   cv2.VideoWriter_fourcc(*video_codec),
                                   fps,
                                   (width, height))
        
        alerts_file = open(os.environ.get("ALERTS_FILE"), "w")

        for frame in tqdm(range(n_frames), total=n_frames):
            ret, img = cap.read()
            if not ret : break

            # full
            pred_df, collision_zone_list, collisions = self.pc_model.detect(img, conf_thres, iop_thres)
            img_ = self.add_annotations(img, pred_df, collision_zone_list, collisions)

            time_label = VideoWriter.get_time_label(frame, fps)
            img_ = VideoWriter.add_info(img_, time=time_label, collision_count=len(collisions))

            out_full.write(img_)

            # alert
            if collisions:
                out_alerts.write(img_)
                alert_label = f"Time: {time_label} - Frame: {frame} - Collisions: {len(collisions)}"
                alerts_file.write(alert_label + "\n")

        out_full.release()
        out_alerts.release()
        cap.release()
        alerts_file.close()
        logger.log_L1("Finished generating video.")

        
    def add_annotations(self, img, pred_df, collision_zone_list, collisions):
        """
        Add annotations from predictions to image.

        @img: image
        @pred_df: prediction dataframe from OD model.
        @collision_zone_list: polygons of collision zones for each vehicle.
        @collisions: list of collisions (person, vehicle and IOP score) for the image.
        """
        img_ = img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness_polygons = 2
        thickness_text = 1
        
        class_map = { v:k for k,v in self.pc_model.od_model.classes.items() }
        people_danger = [ int(col["person_id"]) for col in collisions ]
        
        # draw the collision zones
        for collision_zone_pts in collision_zone_list:
            pts = np.array(collision_zone_pts, np.int32)
            pts = pts.reshape((-1, 1, 2))
            img_ = cv2.polylines(img_, [pts], True, self.color_map["collision_zone"], thickness=thickness_polygons)

        # draw the bboxes
        for _, pred in pred_df.iterrows():
            xmin, ymin, xmax, ymax = pred["xmin"], pred["ymin"], pred["xmax"], pred["ymax"]

            if pred["id"] in people_danger:
                color = self.color_map["person_in_danger"]
            else:
                color = self.color_map[class_map[pred['class']]]

            pt1 = int(xmin), int(ymin)
            pt2 = int(xmax), int(ymax)
            img_ = cv2.rectangle(img_, pt1, pt2, color, thickness=thickness_polygons)

            pt_text = pt1[0] + 5, pt1[1] + 10
            img_ = cv2.putText(img_, class_map[pred['class']], pt_text, font, 0.5, color, thickness=thickness_text)

            confidence_label = f"{pred['confidence']:.2f}"
            pt_conf = pt2[0] + 5, pt2[1] + 10
            img_ = cv2.putText(img_, confidence_label, pt_conf, font, 0.5, color, thickness=thickness_text)

        return img_
    
    @staticmethod
    def add_info(img, time, collision_count):
        """
        Add info as timestamp and amount of collisions in the image.

        @img: image
        @time: timestamp label
        @collision_count: amount of collisions in the image.
        """
        def draw_black_box(img, text, x=0, y=0, w=90, h=20):
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,0), -1)
            img = cv2.putText(img, text, (x + int(w/7), y + int(h/1.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
            return img
        
        img_ = img.copy()
        time_label = f"time: {time}"
        collision_label = f"collisions: {collision_count}"
        img_time = draw_black_box(img_, time_label, x=0, y=0)
        img_col = draw_black_box(img_time, collision_label, x=0, y=20)
    
        return img_col

    @staticmethod
    def get_time_label(frame, fps):
        """
        Calculate time of the frame in the video given.

        @frame: frame of the video.
        @fps: frame per second.
        """
        seconds = frame / fps
        rt = relativedelta(seconds=seconds)
        time_label = f"{int(rt.hours):02}:{int(rt.minutes):02}:{int(rt.seconds):02}"

        return time_label
