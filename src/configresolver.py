import configparser
import logging

from facedetector import FaceDetector

class ConfigResolver:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('./resources/config.ini')

    def get_face_detector(self):
        if not self.config.has_section('FaceDetector'):
            raise ValueError('No FaceDetector configuration!')
        else:
            cascade_classifier_path = self.config.get('FaceDetector', 'cascadeClassifier', fallback='./resources/haarcascade_frontalface_default.xml')
            scaleFactor = self.config.getfloat('FaceDetector', 'scaleFactor', fallback=1.3)
            minNeighbors = self.config.getint('FaceDetector', 'minNeighbors', fallback=4)
            minSize_x = self.config.getint('FaceDetector', 'minSize_x', fallback=40)
            minSize_y = self.config.getint('FaceDetector', 'minSize_y', fallback=40)
            return FaceDetector(cascade_classifier_path, scaleFactor, minNeighbors, (minSize_x, minSize_y))
