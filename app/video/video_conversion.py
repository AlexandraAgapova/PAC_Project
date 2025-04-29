import cv2
import numpy as np


class Video:
    """
    video iteration and frame-by-frame recording
    """
    def __init__(self, path_in, path_out):
        """
        initialize input and output video objects
        """
        self.path_in = path_in
        self.path_out = path_out
        self.cap = cv2.VideoCapture(path_in)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {path_in}")

        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.out = cv2.VideoWriter(path_out, self.fourcc, self.fps, (self.width, self.height))

    def __iter__(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            self.close()
            raise StopIteration

        return frame

    def __len__(self):
        return self.frame_count
    
    def write(self, frame):
        self.out.write(frame)

    def close(self):
        self.cap.release()
        self.out.release()
