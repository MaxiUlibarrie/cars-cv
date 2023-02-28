from images_utils import sqr_to_polygon, IOP

class PreventCollision():

    danger_vehicles = ["car", "motorcycle", "bus"]
    
    def __init__(self, od_model):
        self.od_model = od_model
        
    def detect(self, img, conf_thres, iop_thres):
        """
        Detects objects and collisions in the image.

        @img: image
        @conf_thres: Confidence threshold for the OD model.
        @iop_thres: Intersection over person threshold.
        """
        height = img.shape[0]
        width = img.shape[1]
        pred_df = self.od_model.predict(img, conf_thres)

        vehicles_ids = [ v for k,v in self.od_model.classes.items() if k in PreventCollision.danger_vehicles ]
        vehicles = pred_df[ pred_df["class"].isin(vehicles_ids) ]
        people = pred_df[ pred_df["class"] == self.od_model.classes["person"] ]
        
        collisions = []
        collision_zone_list = [] 
        for _, vehicle in vehicles.iterrows():
            collision_zone = PreventCollision.get_collision_zone(vehicle["xmin"], vehicle["ymin"], \
                                                                 vehicle["xmax"], vehicle["ymax"], \
                                                                 width, height)
            collision_zone_list.append(collision_zone)
            for _, person in people.iterrows():
                person_pts = sqr_to_polygon(person["xmin"], person["ymin"], \
                                            person["xmax"], person["ymax"])
                iop_score = IOP(img, collision_zone, person_pts)
                if iop_score >= iop_thres:
                    collisions.append({ 
                        "vehicle_id" : int(vehicle["id"]), 
                        "person_id" : int(person["id"]),
                        "iop_score" : iop_score
                    })
                    
        return pred_df, collision_zone_list, collisions

    @staticmethod
    def get_collision_zone(xmin, ymin, xmax, ymax, 
                           width, height,
                           width_col_zone = 0.8, flap = 10):
        """
        Gets collision zone from coordinates of the vehicle.

        @xmin, ymin, xmax, ymax: coordinates
        @width, height: size of the image
        @width_col_zone: percentage of the vehicle to apply \
            into the width of the collision zone
        @flap: flaps of bbox of the collision zone
        """
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
    