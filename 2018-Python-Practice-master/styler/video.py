import cv2

from styler.utils import resize


class Video:

    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(self.path)
        self.frames = []

    def __enter__(self):
        if not self.cap.isOpened():
            raise Exception('Cannot open video: {}'.format(self.path))
        return self

    def __len__(self):
        return len(self.frames)

    def read_frames(self, image_h, image_w):
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            if ret == True:
                self.frames.append(resize(frame, image_h, image_w))
            else: break

        return self.frames

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cap.release()
