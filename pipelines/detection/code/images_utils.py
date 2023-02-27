import numpy as np
import cv2


def sqr_to_polygon(xmin, ymin, xmax, ymax): 
    """
    Convert square points into polygon points.

    @
    """
    pts = [
        [xmin, ymin], 
        [xmin + (xmax - xmin), ymin],
        [xmax, ymax], 
        [xmin, ymin + (ymax - ymin)]
    ]
    
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