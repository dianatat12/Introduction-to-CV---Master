import numpy as np
import cv2 as cv

class KPDescriptor:
    def calculate(self, img):
        pass



class SURF_des(KPDescriptor):
    def __init__(self):
        pass
    
    def calculate_keypoints(self, img):
        #im = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        surf = cv.xfeatures2d.SURF_create(400)
        kp, desc = surf.detectAndCompute(img,None)
        return desc
    
    def calculate(self, img):
        return self.calculate_keypoints(img)
 
        
class SIFT_des(KPDescriptor):
    def __init__(self):
        pass

    def calculate_keypoints(self, img):
        #im = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        sift = cv.SIFT_create(nfeatures=3500)
        kp, desc = sift.detectAndCompute(img,None)
        return desc
    
    def calculate(self, img):
        return self.calculate_keypoints(img)
    
    
class ORB_des(KPDescriptor):
    def __init__(self):
        pass

    def calculate_keypoints(self, img):
        #im = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        orb = cv.ORB_create(nfeatures=3500)
        kp, desc = orb.detectAndCompute(img,None)
        return desc
    
    def calculate(self, img):
        return self.calculate_keypoints(img)
        

class FAST_des(KPDescriptor):
    def __init__(self):
        pass

    def calculate_keypoints(self, img):
        fast = cv.FastFeatureDetector_create()
        kp = fast.detect(img, None)

        # Compute descriptors using ORB
        orb = cv.ORB_create(nfeatures=3500)
        kp, desc = orb.detectAndCompute(img,None)
        return desc
    
    def calculate(self, img):
        return self.calculate_keypoints(img)
    

