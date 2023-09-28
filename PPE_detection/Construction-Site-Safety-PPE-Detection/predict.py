
# All imports used
from ultralytics import YOLO
import torch
import cv2
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import subprocess
import yaml
import glob
import re
import os
import warnings
warnings.filterwarnings("ignore")


sns.set_palette('Set3')

# import IPython.display as display
# from IPython.display import Video


# CFG stores all paths and parameters that might be useful
class CFG(object):
    # inference: use any pretrained or custom model
    # WEIGHTS = 'yolov8x.pt' # yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    WEIGHTS = 'models/best.pt'

    CONFIDENCE = 0.35
    CONFIDENCE_INT = int(round(CONFIDENCE * 100, 0))

    # ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask','NO-Safety Vest', 'Person', 'Safety Cone','Safety Vest', 'machinery', 'vehicle']
    CLASSES_TO_DETECT = [0, 1, 2, 3]

    VERTICES_POLYGON = np.array(
        [[200, 720], [0, 700], [500, 620], [990, 690], [820, 720]])

    EXP_NAME = 'ppe'

    # just some video examples
    VID_001 = 'assets/test.mp4'

    # choose filepath to make inference on (image or video)
    PATH_TO_INFER_ON = VID_001
    EXT = PATH_TO_INFER_ON.split('.')[-1]  # get file extension
    FILENAME_TO_INFER_ON = PATH_TO_INFER_ON.split(
        '/')[-1].split('.')[0]  # get filename

    # paths
    ROOT_DIR = ''
    OUTPUT_DIR = ''


# This is the instance of the custom model fine tuned on the publicly available yolov8 model, trained on a PPE dataset from roblytics
class Video(CFG):
    def __init__(self, path):
        self.path = path

    def predict(self):
        """
        This returns the prediction results
        params:
        -------
        """
        model = YOLO(CFG.WEIGHTS)
        results = model.predict(
            source=self.path,
            save=True,
            classes=CFG.CLASSES_TO_DETECT,
            conf=CFG.CONFIDENCE,
            save_txt=True,
            save_conf=True,
            show=True,
            device="cpu",
            # stream=True,
        )
        return results


# Helper function to play a video on cv2 (Uninstanced)
def playvid(vid):

    cap = cv2.VideoCapture(vid)

    height = 1080
    width = 1920

    cap = cv2.VideoCapture(vid)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)

    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        resize = cv2.resize(frame, (height, width))

        if ret == True:

            # Display the resulting frame
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

        # When everything done, release the video capture object
    cap.release()


# Function when live camera stream is used
def cameraseq(camera):
    """
    This function is used to play a live camera stream
    params:
    ------- camera : cv2 camera object (Eg: camera = cv2.imread(0))
    """

    # To count frames
    img_counter = 0
    while True:
        # Frame read from camera
        ret, frame = camera.read()

        if not ret:
            print("Failed to retrieve frame")
            break
        # Display the frame for viewing (This creates a live video stream window useful for debugging)
        cv2.imshow("test", frame)

        # To keep track of keystrokes
        k = cv2.waitKey(1)

        # esc key
        if k % 256 == 27:
            print("Closing")
            break
        # space key
        elif k % 256 == 32:

            # path to where output frames are to be stored
            img_path = f"output/opencv_frame{img_counter}.png"

            cv2.imwrite(img_path, frame)
            model = Video(img_path)
            results = model.predict()
            img_counter += 1
            # Display the scanned frame with the predicted classes and bounding boxes

            for result in results:
                boxes = result.boxes
                probs = result.probs
                fig = plt.figure()
                plt.imshow(result.plot())
                plt.savefig(f"output/opencv_frame_output{img_counter}.png")
            print(boxes)
            print(probs)

    camera.release()


def vidseq(vid):
    """
    This function is used to process a video
    -------parms: vid = path to video
    Results are stored in working/runs/detect/predict
    """

    model = Video(vid)
    results = model.predict()


def app():

    path = "assets/production_id_3960165 (540p).mp4"
    # production_id_3960165 (540p).mp4"
    # path = 0
    vid = cv2.VideoCapture(path)
    # playvid(path)

    if not (vid.isOpened()):
        print("Error: Could not open video file")
        exit()

    if path == 0:
        cameraseq(vid)
    else:
        vidseq(path)


if __name__ == '__main__':
    app()
