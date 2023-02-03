import torch

class Yolo():

    def __init__(self, weights, custom=True):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', weights) 

    def predict_bboxes(self, img):
        results = self.model(img)
        labels, coord_thres = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
        return labels, coord_thres

    def predict_bboxes_as_df(self, img):
        results = self.model(img)
        return results.pandas().xyxy[0]