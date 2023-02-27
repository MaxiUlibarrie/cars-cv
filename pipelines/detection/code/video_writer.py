import cv2
import numpy as np
from tqdm import tqdm

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
        
    def add_annotations(self, img, pred_df, collision_zone_list, collisions):
        font = cv2.FONT_HERSHEY_TRIPLEX
        img_ = img.copy()
        
        class_map = { v:k for k,v in self.pc_model.od_model.classes.items() }
        people_danger = [ int(col["person_id"]) for col in collisions ]
        
        # draw the collision zones
        for collision_zone_pts in collision_zone_list:
            pts = np.array(collision_zone_pts, np.int32)
            pts = pts.reshape((-1, 1, 2))
            img_ = cv2.polylines(img_, [pts], True, self.color_map["collision_zone"], 4)

        # draw the bboxes
        for _, pred in pred_df.iterrows():
            xmin, ymin, xmax, ymax = pred["xmin"], pred["ymin"], pred["xmax"], pred["ymax"]

            if pred["id"] in people_danger:
                color = self.color_map["person_in_danger"]
            else:
                color = self.color_map[class_map[pred['class']]]

            pt1 = int(xmin), int(ymin)
            pt2 = int(xmax), int(ymax)
            img_ = cv2.rectangle(img_, pt1, pt2, color, 3)

            pt_text = pt1[0] + 5, pt1[1] + 10
            img_ = cv2.putText(img_, class_map[pred['class']], pt_text, font, 0.5, color)

            confidence_label = f"{pred['confidence']:.2f}"
            pt_conf = pt2[0] + 5, pt2[1] + 10
            img_ = cv2.putText(img_, confidence_label, pt_conf, font, 0.5, color)

        return img_
    
    def generate_video(self, video_path, video_output,
                       width = 640, 
                       height = 480,
                       conf_thres = 0.3,
                       iop_thres = 0.5,
                       video_codec = "mp4v"):
        
        logger.log_L1("Started to generate video.")
        cap = cv2.VideoCapture(video_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(video_output,
            cv2.VideoWriter_fourcc(*video_codec),
            fps,
            (width, height))
        
        for frame in tqdm(range(n_frames), total=n_frames):
            ret, img = cap.read()
            if not ret : break

            pred_df, collision_zone_list, collisions = self.pc_model.detect(img, conf_thres, iop_thres)
            img_ = self.add_annotations(img, pred_df, collision_zone_list, collisions)
            out.write(img_)

        out.release()
        cap.release()
        logger.log_L1("Finished generating video.")
