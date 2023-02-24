import numpy as np
import cv2

class PreventCollision():
    
    def __init__(self, od_model):
        self.od_model = od_model
        
    def detect(self, img, conf_thres, iop_thres):
        height = img.shape[0]
        width = img.shape[1]
        pred_df = self.od_model.predict(img, conf_thres)
        cars = pred_df[pred_df["class"] == self.od_model.classes["car"]]
        people = pred_df[pred_df["class"] == self.od_model.classes["person"]]
        
        collisions = []
        collision_zone_list = [] 
        for _, car in cars.iterrows():
            collision_zone = PreventCollision.get_collision_zone(car["xmin"], car["ymin"], \
                                                                 car["xmax"], car["ymax"], \
                                                                 width, height)
            collision_zone_list.append(collision_zone)
            for _, person in people.iterrows():
                person_pts = PreventCollision.sqr_to_polygon((person["xmin"], person["ymin"]), \
                                                             (person["xmax"], person["ymax"]))
                iop_score = PreventCollision.IOP(img, collision_zone, person_pts)
                if iop_score >= iop_thres:
                    collisions.append({ 
                        "car_id" : int(car["id"]), 
                        "person_id" : int(person["id"]),
                        "iop_score" : iop_score
                    })
                    
        return pred_df, collision_zone_list, collisions

    def get_collision_zone(xmin, ymin, xmax, ymax, width, height,
                           width_col_zone = 0.8, flap = 10):
        length = (xmax - xmin) * width_col_zone
        pt1 = [xmin, ymin]
        pt2 = [xmin, ymin + (ymax-ymin)]
        pt3 = [pt2[0] - length, pt2[1] + flap]
        pt4 = [pt1[0] - length, pt1[1] - flap]
        pts = [pt1, pt2, pt3, pt4]
        
        for pt in pts:
            if pt[0] >= width : pt[0] = width - 1
            if pt[1] >= height : pt[1] = height - 1
            if pt[0] < 0 : pt[0] = 0
            if pt[1] < 0 : pt[1] = 0
        
        pts = [ [ int(pt[0]), int(pt[1]) ] for pt in pts ]

        return pts
    
    def sqr_to_polygon(pt1, pt2): 
        pts = [list(pt1), [pt1[0] + (pt2[0] - pt1[0]), pt1[1]], \
               list(pt2), [pt1[0], pt1[1] + (pt2[1] - pt1[1])]]
        
        pts = [ [ int(pt[0]), int(pt[1]) ] for pt in pts ]
            
        return pts
    
    # Intersection Over Person
    def IOP(img, collision_pts, person_pts):
        def mask(img, bb_points):
            stencil = np.zeros(img.shape).astype(img.dtype)
            contours = [np.array(bb_points)]
            color = [255, 255, 255]
            result = cv2.fillPoly(stencil, contours, color)
            return cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        collision_mask = mask(img, collision_pts)
        person_mask = mask(img, person_pts)

        intersection = np.logical_and(collision_mask, person_mask)
        person_region = np.logical_and(person_mask, person_mask)
        iop_score = np.sum(intersection) / np.sum(person_region)
        
        return iop_score
    