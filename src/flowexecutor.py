import logging
import cv2

from emotionrecognizer import EmotionRecognizer
from configresolver import ConfigResolver


class FlowExecutor:
    def __init__(self):
        self.config_resolver = ConfigResolver()

        self.face_detector = self.config_resolver.get_face_detector()
        self.emotion_recognizer = EmotionRecognizer()

    # crop image of size
    def __crop(self, image, p1, p2):
        return image[p1[1]:p2[1], p1[0]:p2[0], :]

    # add bounding box with emotions label
    def __add_labeled_bounding_box(self, image, predicted_emotion, pt1, pt2):
        cv2.rectangle(image, pt1, pt2, (0, 255, 255))
        cv2.putText(image, predicted_emotion, (pt1[0], pt1[1] + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))

    def execute(self, image):
        image_copy = image.copy()
        
        faces = self.face_detector.detect_faces(image_copy)  # faces coordinates

        from collections import defaultdict
        group_emotions = defaultdict(float)
        avg_emo = 'Unknown'
        if faces is not None and len(faces) > 0:
            for (x, y, w, h) in faces:
                face = self.__crop(image_copy, (x, y), (x + w, y + h))
                if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                    cv2.imshow('Face', face)
                height, width = face.shape[:2]
                if height > 0 and width > 0:
                    logging.info('Start')
                    predicted_emotion, emo_map = self.emotion_recognizer.recognize(face)
                    print(emo_map)
                    logging.info('Emotion: %s', predicted_emotion)
                    self.__add_labeled_bounding_box(image, predicted_emotion, (x, y), (x + w, y + h))  # pass original image for bounding
                    for emo, val in emo_map.items():
                        group_emotions[emo] += val
            print(group_emotions)
            avg_emo, val = max(group_emotions.items(), key=lambda x: x[1])
        else:
            logging.warning('No face was found!')

        return image, avg_emo
