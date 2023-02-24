import numpy as np
import pandas as pd
import torch
from abc import ABC, abstractmethod
from common import Config

config = Config()

class OD(ABC):

    @abstractmethod
    def predict(*args, **kwargs):
        pass

class Yolov5(OD):
    
    classes = config.classes
    
    def __init__(self, weights):
        if weights in ["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x"]:
            self.model = torch.hub.load('ultralytics/yolov5', weights)
        else:
            # custom 
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', weights)
        
    def predict(self, img, conf_thres):
        results = self.model(img)
        pred_df = pd.DataFrame()
        pred_df["xmin"] = results.xyxy[0][:,0].numpy().astype(np.int16)
        pred_df["ymin"] = results.xyxy[0][:,1].numpy().astype(np.int16)
        pred_df["xmax"] = results.xyxy[0][:,2].numpy().astype(np.int16)
        pred_df["ymax"] = results.xyxy[0][:,3].numpy().astype(np.int16)
        pred_df["confidence"] = results.xyxy[0][:,4].numpy().astype(np.float32)
        pred_df["class"] = results.xyxy[0][:,5].numpy().astype(np.int8)

        pred_df = pred_df[pred_df["class"].isin(Yolov5.classes.values())]
        pred_df = pred_df[pred_df['confidence'] >= conf_thres] 
        pred_df["id"] = np.arange(len(pred_df), dtype=np.int16)
        
        return pred_df
    