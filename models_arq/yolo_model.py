import torch
from abc import ABC, abstractmethod

from useful import Config

class Yolov5(ABC):

    @abstractmethod
    def __ini__(self, weights):
        pass

    @abstractmethod
    def predict_as_df(self, img, conf_thres):
        pass

class Yolov5_custom(Yolov5):

    def __init__(self, weights):
        super(Yolov5_custom, self).__init__(weights)
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', weights) 

    def predict_as_df(self, img, conf_thres):
        results = self.model(img)
        df = results.pandas().xyxy[0]
        df = df[df['confidence'] >= conf_thres]
        return df

class Yolov5_standard(Yolov5):

    def __init__(self, weights):
        super(Yolov5_standard, self).__init__(weights)
        self.model = torch.hub.load('ultralytics/yolov5', weights)

        config = Config()
        self.classes = list(config.classes.values())


    def predict_as_df(self, img, conf_thres):
        results = self.model(img)
        df = results.pandas().xyxy[0]
        df = df[df["name"].isin(self.classes)]
        df = df[df['confidence'] >= conf_thres]
        return df 
